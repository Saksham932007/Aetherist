#!/usr/bin/env python3
"""Script for optimizing Aetherist models for production deployment."""

import argparse
import logging
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.deployment.model_optimizer import ModelOptimizer, OptimizationConfig
from src.config.config_manager import ConfigManager
from src.models.generator import AetheristGenerator
from src.models.discriminator import AetheristDiscriminator

def main():
    parser = argparse.ArgumentParser(description="Optimize Aetherist models for production")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                       help="Path to model configuration file")
    parser.add_argument("--generator-checkpoint", type=str, required=True,
                       help="Path to generator checkpoint")
    parser.add_argument("--discriminator-checkpoint", type=str,
                       help="Path to discriminator checkpoint")
    parser.add_argument("--output-dir", type=str, default="deployments/optimized_models",
                       help="Output directory for optimized models")
    parser.add_argument("--quantize", action="store_true",
                       help="Enable model quantization")
    parser.add_argument("--export-onnx", action="store_true",
                       help="Export models to ONNX format")
    parser.add_argument("--export-torchscript", action="store_true",
                       help="Export models to TorchScript format")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmarks on optimized models")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size for optimization")
    parser.add_argument("--image-size", type=int, nargs=2, default=[512, 512],
                       help="Image size for optimization")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
        
        # Create optimization configuration
        optimization_config = OptimizationConfig(
            quantize=args.quantize,
            export_onnx=args.export_onnx,
            export_torchscript=args.export_torchscript,
            batch_size=args.batch_size,
            image_size=tuple(args.image_size)
        )
        
        # Initialize optimizer
        optimizer = ModelOptimizer(optimization_config)
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and optimize generator
        logger.info("Loading generator model...")
        try:
            generator = AetheristGenerator(config.model)
            generator.load_state_dict(torch.load(args.generator_checkpoint, map_location='cpu'))
            generator.eval()
            
            logger.info("Optimizing generator...")
            gen_results = optimizer.optimize_generator(generator, output_dir / "generator")
            logger.info(f"Generator optimization completed: {gen_results}")
            
        except Exception as e:
            logger.error(f"Generator optimization failed: {e}")
            # Create placeholder for missing model
            gen_results = {"error": str(e)}
            
        # Load and optimize discriminator if checkpoint provided
        disc_results = {}
        if args.discriminator_checkpoint:
            logger.info("Loading discriminator model...")
            try:
                discriminator = AetheristDiscriminator(config.model)
                discriminator.load_state_dict(torch.load(args.discriminator_checkpoint, map_location='cpu'))
                discriminator.eval()
                
                logger.info("Optimizing discriminator...")
                disc_results = optimizer.optimize_discriminator(discriminator, output_dir / "discriminator")
                logger.info(f"Discriminator optimization completed: {disc_results}")
                
            except Exception as e:
                logger.error(f"Discriminator optimization failed: {e}")
                disc_results = {"error": str(e)}
                
        # Run benchmarks if requested
        benchmark_results = {}
        if args.benchmark:
            logger.info("Running benchmarks...")
            try:
                import torch
                sample_latent = torch.randn(args.batch_size, 512)
                sample_image = torch.randn(args.batch_size, 3, *args.image_size)
                
                if "error" not in gen_results:
                    gen_benchmark = optimizer.benchmark_models(gen_results, sample_latent)
                    benchmark_results["generator"] = gen_benchmark
                    
                if "error" not in disc_results:
                    disc_benchmark = optimizer.benchmark_models(disc_results, sample_image)
                    benchmark_results["discriminator"] = disc_benchmark
                    
                logger.info(f"Benchmark results: {benchmark_results}")
                
            except Exception as e:
                logger.error(f"Benchmarking failed: {e}")
                benchmark_results = {"error": str(e)}
                
        # Save optimization report
        logger.info("Saving optimization report...")
        report_path = output_dir / "optimization_report.json"
        
        # Combine results for report
        all_results = {**gen_results}
        if disc_results:
            all_results.update({f"discriminator_{k}": v for k, v in disc_results.items()})
            
        optimizer.save_optimization_report(all_results, benchmark_results, report_path)
        
        logger.info(f"Optimization completed successfully!")
        logger.info(f"Optimized models saved to: {output_dir}")
        logger.info(f"Report saved to: {report_path}")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    # Import torch here to avoid import errors if not available
    try:
        import torch
    except ImportError:
        print("Error: PyTorch is not installed. Please install it to run model optimization.")
        print("Run: pip install torch torchvision")
        sys.exit(1)
        
    main()
