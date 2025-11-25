#!/usr/bin/env python3
"""
Command-line interface for Aetherist inference.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import torch
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.inference.inference_pipeline import AetheristInferencePipeline, BatchInferencePipeline


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using trained Aetherist model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--model-path", "-m", 
        type=str, 
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for inference"
    )
    parser.add_argument(
        "--half-precision", 
        action="store_true",
        help="Use half precision (FP16) for faster inference"
    )
    
    # Generation arguments
    parser.add_argument(
        "--num-samples", "-n",
        type=int, 
        default=4,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int, 
        default=None,
        help="Batch size for generation (default: process all at once)"
    )
    parser.add_argument(
        "--resolution", "-r",
        type=int, 
        default=None,
        help="Output image resolution (default: model's native resolution)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int, 
        default=None,
        help="Random seed for reproducible generation"
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--output-dir", "-o",
        type=str, 
        default="./generated_images",
        help="Output directory for generated images"
    )
    parser.add_argument(
        "--prefix",
        type=str, 
        default="aetherist",
        help="Filename prefix for generated images"
    )
    parser.add_argument(
        "--format",
        type=str, 
        default="PNG",
        choices=["PNG", "JPEG", "WEBP"],
        help="Output image format"
    )
    
    # Advanced options
    parser.add_argument(
        "--latent-codes",
        type=str, 
        default=None,
        help="Path to custom latent codes (.npy or .json file)"
    )
    parser.add_argument(
        "--camera-poses",
        type=str, 
        default=None,
        help="Path to custom camera poses (.npy file)"
    )
    parser.add_argument(
        "--save-latents",
        action="store_true",
        help="Save used latent codes to output directory"
    )
    parser.add_argument(
        "--save-cameras",
        action="store_true",
        help="Save used camera poses to output directory"
    )
    parser.add_argument(
        "--return-triplane",
        action="store_true",
        help="Save triplane representations"
    )
    
    # Interpolation mode
    parser.add_argument(
        "--interpolate",
        action="store_true",
        help="Generate interpolation between two latent codes"
    )
    parser.add_argument(
        "--interpolation-steps",
        type=int, 
        default=10,
        help="Number of interpolation steps"
    )
    parser.add_argument(
        "--interpolation-method",
        type=str, 
        default="slerp",
        choices=["linear", "slerp"],
        help="Interpolation method"
    )
    
    # Utility flags
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show model information and exit"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize inference pipeline
        logger.info("Initializing inference pipeline...")
        pipeline = AetheristInferencePipeline(
            model_path=args.model_path,
            device=args.device,
            half_precision=args.half_precision,
        )
        
        # Show model info if requested
        if args.info:
            info = pipeline.get_model_info()
            print("\n=== Model Information ===")
            for key, value in info.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for k, v in value.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"{key}: {value}")
            return
        
        # Load custom inputs if provided
        latent_codes = None
        camera_poses = None
        
        if args.latent_codes:
            logger.info(f"Loading custom latent codes from {args.latent_codes}")
            if args.latent_codes.endswith('.npy'):
                latent_codes = torch.from_numpy(np.load(args.latent_codes))
            elif args.latent_codes.endswith('.json'):
                with open(args.latent_codes, 'r') as f:
                    data = json.load(f)
                latent_codes = torch.tensor(data)
            else:
                raise ValueError("Latent codes must be .npy or .json file")
        
        if args.camera_poses:
            logger.info(f"Loading custom camera poses from {args.camera_poses}")
            camera_poses = torch.from_numpy(np.load(args.camera_poses))
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate images
        if args.interpolate:
            # Interpolation mode
            if latent_codes is None or latent_codes.shape[0] < 2:
                logger.error("Interpolation mode requires at least 2 latent codes")
                return
            
            logger.info(f"Generating {args.interpolation_steps} interpolation steps")
            result = pipeline.interpolate(
                start_latent=latent_codes[0],
                end_latent=latent_codes[1],
                steps=args.interpolation_steps,
                camera_pose=camera_poses[0] if camera_poses is not None else None,
                interpolation_method=args.interpolation_method,
            )
        else:
            # Standard generation mode
            if args.num_samples > 16 or args.batch_size is not None:
                # Use batch pipeline for large generations
                batch_pipeline = BatchInferencePipeline(
                    pipeline,
                    max_batch_size=args.batch_size or 8,
                    enable_progress=not args.no_progress,
                )
                
                logger.info(f"Generating {args.num_samples} images in batches...")
                batch_result = batch_pipeline.generate_large_batch(
                    num_samples=args.num_samples,
                    output_dir=output_dir,
                    latent_codes=latent_codes,
                    camera_poses=camera_poses,
                    resolution=args.resolution,
                    return_triplane=args.return_triplane,
                    seed=args.seed,
                )
                
                logger.info(f"Generated {batch_result['num_generated']} images in {batch_result['total_time']:.2f}s")
                logger.info(f"Average time per sample: {batch_result['avg_time_per_sample']:.2f}s")
                return
            
            else:
                # Single batch generation
                logger.info(f"Generating {args.num_samples} images...")
                result = pipeline.generate(
                    num_samples=args.num_samples,
                    latent_codes=latent_codes,
                    camera_poses=camera_poses,
                    resolution=args.resolution,
                    batch_size=args.batch_size,
                    return_triplane=args.return_triplane,
                    seed=args.seed,
                )
        
        # Save images
        logger.info(f"Saving images to {output_dir}")
        saved_paths = pipeline.save_images(
            result["images"],
            output_dir,
            prefix=args.prefix,
            format=args.format,
        )
        
        # Save additional outputs if requested
        if args.save_latents:
            latent_path = output_dir / f"{args.prefix}_latent_codes.npy"
            np.save(latent_path, result["latent_codes"].numpy())
            logger.info(f"Saved latent codes to {latent_path}")
        
        if args.save_cameras:
            camera_path = output_dir / f"{args.prefix}_camera_poses.npy"
            np.save(camera_path, result["camera_poses"].numpy())
            logger.info(f"Saved camera poses to {camera_path}")
        
        if args.return_triplane and "triplanes" in result:
            triplane_path = output_dir / f"{args.prefix}_triplanes.npy"
            np.save(triplane_path, result["triplanes"].numpy())
            logger.info(f"Saved triplanes to {triplane_path}")
        
        # Save generation metadata
        metadata = {
            "model_path": args.model_path,
            "num_samples": len(result["images"]),
            "generation_config": {
                "seed": args.seed,
                "resolution": args.resolution,
                "batch_size": args.batch_size,
                "device": args.device,
                "half_precision": args.half_precision,
            },
            "model_info": pipeline.get_model_info(),
            "saved_files": saved_paths,
        }
        
        if args.interpolate:
            metadata["interpolation"] = {
                "steps": args.interpolation_steps,
                "method": args.interpolation_method,
            }
        
        metadata_path = output_dir / f"{args.prefix}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
        
        logger.info(f"Generation complete! Saved {len(saved_paths)} images to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()