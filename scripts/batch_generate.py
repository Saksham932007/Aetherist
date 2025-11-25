#!/usr/bin/env python3
"""Batch generation script for Aetherist.

Generates large batches of images efficiently using the batch processor.
"""

import argparse
import logging
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    import numpy as np
    from PIL import Image
    from src.batch.batch_processor import BatchProcessor, BatchConfig
    from src.inference.inference_pipeline import AetheristInferencePipeline
    from src.config.config_manager import ConfigManager
    from src.utils.camera_utils import CameraPoseGenerator
    GENERATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    GENERATION_AVAILABLE = False

def image_generation_processor(generation_params: Dict[str, Any], 
                              pipeline: AetheristInferencePipeline,
                              camera_generator: CameraPoseGenerator,
                              **kwargs) -> Dict[str, Any]:
    """Process image generation job."""
    try:
        # Extract parameters
        num_samples = generation_params.get('num_samples', 1)
        resolution = generation_params.get('resolution', 512)
        seed = generation_params.get('seed')
        latent_codes = generation_params.get('latent_codes')
        camera_params = generation_params.get('camera_params')
        
        # Generate latent codes if not provided
        if latent_codes is None:
            if seed is not None:
                torch.manual_seed(seed)
            latent_codes = torch.randn(num_samples, 512)
            
        # Generate camera parameters if not provided
        if camera_params is None:
            camera_params = [camera_generator.sample_random_pose() for _ in range(num_samples)]
            
        # Generate images
        with torch.no_grad():
            results = pipeline.generate(
                latent_codes=latent_codes,
                camera_params=camera_params,
                output_size=(resolution, resolution)
            )
            
        # Save images if output paths provided
        output_paths = []
        output_dir = kwargs.get('output_dir')
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            for i, image in enumerate(results['images']):
                if isinstance(image, torch.Tensor):
                    # Convert tensor to PIL Image
                    image_np = image.detach().cpu().numpy()
                    if image_np.ndim == 3 and image_np.shape[0] == 3:
                        image_np = image_np.transpose(1, 2, 0)
                    image_np = (image_np * 255).astype(np.uint8)
                    image_pil = Image.fromarray(image_np)
                else:
                    image_pil = image
                    
                # Save image
                filename = f"generated_{kwargs.get('job_id', 'unknown')}_{i:04d}.png"
                image_path = output_dir / filename
                image_pil.save(image_path)
                output_paths.append(str(image_path))
                
        return {
            'num_generated': len(results['images']),
            'resolution': resolution,
            'output_paths': output_paths,
            'generation_params': generation_params,
            'camera_params': camera_params,
            'seed': seed
        }
        
    except Exception as e:
        raise RuntimeError(f"Image generation failed: {str(e)}")

def create_generation_jobs(num_images: int,
                          images_per_job: int,
                          resolution: int,
                          start_seed: Optional[int] = None,
                          camera_preset: Optional[str] = None) -> List[Dict[str, Any]]:
    """Create list of generation job parameters."""
    jobs = []
    current_seed = start_seed
    
    remaining_images = num_images
    job_index = 0
    
    while remaining_images > 0:
        batch_size = min(images_per_job, remaining_images)
        
        job_params = {
            'num_samples': batch_size,
            'resolution': resolution,
            'seed': current_seed + job_index if current_seed is not None else None,
            'job_index': job_index
        }
        
        # Add camera preset if specified
        if camera_preset:
            job_params['camera_preset'] = camera_preset
            
        jobs.append(job_params)
        remaining_images -= batch_size
        job_index += 1
        
    return jobs

def main():
    parser = argparse.ArgumentParser(description="Generate images in batches using Aetherist")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                       help="Path to model configuration")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--num-images", type=int, default=100,
                       help="Total number of images to generate")
    parser.add_argument("--images-per-job", type=int, default=8,
                       help="Number of images per batch job")
    parser.add_argument("--resolution", type=int, default=512,
                       help="Output image resolution")
    parser.add_argument("--start-seed", type=int,
                       help="Starting seed for reproducible generation")
    parser.add_argument("--output-dir", type=str, default="outputs/batch_generated",
                       help="Output directory for generated images")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda, cpu, auto)")
    parser.add_argument("--camera-preset", type=str, 
                       choices=["random", "front_view", "side_view", "top_view"],
                       default="random", help="Camera pose preset")
    parser.add_argument("--save-metadata", action="store_true",
                       help="Save generation metadata")
    parser.add_argument("--monitoring", action="store_true",
                       help="Enable performance monitoring")
    parser.add_argument("--timeout", type=int, default=300,
                       help="Timeout per job in seconds")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    if not GENERATION_AVAILABLE:
        logger.error("Required dependencies not available for batch generation")
        sys.exit(1)
        
    try:
        logger.info("Starting batch image generation")
        logger.info(f"Total images: {args.num_images}")
        logger.info(f"Images per job: {args.images_per_job}")
        logger.info(f"Resolution: {args.resolution}")
        logger.info(f"Output directory: {args.output_dir}")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration and initialize pipeline
        logger.info("Loading model...")
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
        
        pipeline = AetheristInferencePipeline(config)
        pipeline.load_checkpoint(args.checkpoint)
        
        camera_generator = CameraPoseGenerator()
        
        # Setup batch processor
        batch_config = BatchConfig(
            max_workers=args.max_workers,
            batch_size=args.images_per_job,
            timeout_seconds=args.timeout,
            output_dir=str(output_dir),
            save_intermediate=args.save_metadata
        )
        
        processor = BatchProcessor(batch_config)
        
        # Register image generation processor
        def generation_wrapper(generation_params: Dict[str, Any], **kwargs) -> Dict[str, Any]:
            return image_generation_processor(
                generation_params, pipeline, camera_generator,
                output_dir=output_dir, **kwargs
            )
            
        processor.register_processor("image_generation", generation_wrapper)
        
        # Create generation jobs
        logger.info("Creating generation jobs...")
        jobs = create_generation_jobs(
            args.num_images,
            args.images_per_job,
            args.resolution,
            args.start_seed,
            args.camera_preset
        )
        
        logger.info(f"Created {len(jobs)} batch jobs")
        
        # Start monitoring if requested
        monitor = None
        if args.monitoring:
            from src.monitoring.system_monitor import SystemMonitor
            monitor = SystemMonitor()
            monitor.start_monitoring()
            
        # Start processing
        start_time = time.time()
        
        with processor:
            # Add all jobs
            job_ids = []
            for i, job_params in enumerate(jobs):
                job_id = processor.add_job("image_generation", job_params)
                job_ids.append(job_id)
                
            logger.info(f"Added {len(job_ids)} jobs to processing queue")
            
            # Wait for completion with progress updates
            completed_count = 0
            last_progress_time = time.time()
            
            while True:
                summary = processor.get_processing_summary()
                new_completed = summary["total_completed"]
                
                if new_completed > completed_count:
                    completed_count = new_completed
                    progress = (completed_count / len(jobs)) * 100
                    elapsed = time.time() - start_time
                    
                    logger.info(f"Progress: {completed_count}/{len(jobs)} jobs ({progress:.1f}%) - "
                               f"Elapsed: {elapsed:.1f}s - Success rate: {summary['success_rate']:.2%}")
                               
                # Check if all jobs are done
                if completed_count + summary["total_failed"] >= len(jobs):
                    break
                    
                # Progress update every 30 seconds
                if time.time() - last_progress_time > 30:
                    logger.info(f"Status: {summary['total_completed']} completed, "
                               f"{summary['total_failed']} failed, {summary['queue_size']} queued")
                    last_progress_time = time.time()
                    
                time.sleep(1)
                
        total_time = time.time() - start_time
        final_summary = processor.get_processing_summary()
        
        # Stop monitoring
        if monitor:
            monitor.stop_monitoring()
            
        # Results summary
        logger.info("\n=== Generation Complete ===")
        logger.info(f"Total time: {total_time:.2f} seconds")
        logger.info(f"Successfully generated: {final_summary['total_completed']} batches")
        logger.info(f"Failed batches: {final_summary['total_failed']}")
        logger.info(f"Success rate: {final_summary['success_rate']:.2%}")
        logger.info(f"Average batch time: {final_summary['avg_processing_time']:.2f} seconds")
        
        images_generated = final_summary['total_completed'] * args.images_per_job
        if images_generated > 0:
            images_per_second = images_generated / total_time
            logger.info(f"Images generated: ~{images_generated}")
            logger.info(f"Generation rate: {images_per_second:.2f} images/second")
            
        # Export results
        logger.info("\n=== Exporting Results ===")
        results_file = output_dir / "batch_generation_results.json"
        processor.export_results(results_file)
        logger.info(f"Results exported to {results_file}")
        
        # Save generation summary
        generation_summary = {
            "total_images_requested": args.num_images,
            "images_per_job": args.images_per_job,
            "resolution": args.resolution,
            "start_seed": args.start_seed,
            "camera_preset": args.camera_preset,
            "total_batches": len(jobs),
            "successful_batches": final_summary['total_completed'],
            "failed_batches": final_summary['total_failed'],
            "total_time_seconds": total_time,
            "avg_batch_time_seconds": final_summary['avg_processing_time'],
            "estimated_images_generated": images_generated,
            "generation_rate_ips": images_per_second if 'images_per_second' in locals() else 0,
            "checkpoint_used": args.checkpoint,
            "config_used": args.config,
            "timestamp": time.time()
        }
        
        summary_file = output_dir / "generation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(generation_summary, f, indent=2)
            
        logger.info(f"Generation summary saved to {summary_file}")
        logger.info(f"\nGenerated images saved to: {output_dir}")
        
        if final_summary['total_failed'] > 0:
            logger.warning(f"Warning: {final_summary['total_failed']} batches failed. "
                          "Check logs for details.")
            
    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
        
if __name__ == "__main__":
    main()
