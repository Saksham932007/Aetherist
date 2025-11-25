#!/usr/bin/env python3
"""Advanced model analysis script for Aetherist.

Provides comprehensive analysis of model architecture, performance,
and quality metrics.
"""

import argparse
import logging
import sys
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    import numpy as np
    from src.monitoring.model_analyzer import ModelAnalyzer
    from src.monitoring.system_monitor import SystemMonitor
    from src.models.generator import AetheristGenerator
    from src.models.discriminator import AetheristDiscriminator
    from src.config.config_manager import ConfigManager
    ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    ANALYSIS_AVAILABLE = False

def main():
    parser = argparse.ArgumentParser(description="Analyze Aetherist model performance")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                       help="Path to model configuration")
    parser.add_argument("--checkpoint", type=str,
                       help="Path to model checkpoint")
    parser.add_argument("--model-type", type=str, choices=["generator", "discriminator", "both"],
                       default="both", help="Which model(s) to analyze")
    parser.add_argument("--analysis-type", type=str, 
                       choices=["architecture", "performance", "quality", "all"],
                       default="all", help="Type of analysis to perform")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8],
                       help="Batch sizes for performance testing")
    parser.add_argument("--num-samples", type=int, default=16,
                       help="Number of samples for quality analysis")
    parser.add_argument("--output-dir", type=str, default="outputs/analysis",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda, cpu, auto)")
    parser.add_argument("--save-visualizations", action="store_true",
                       help="Save analysis visualizations")
    parser.add_argument("--compare-optimized", action="store_true",
                       help="Compare with optimized model versions")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    if not ANALYSIS_AVAILABLE:
        logger.error("Required dependencies not available for analysis")
        sys.exit(1)
        
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting model analysis...")
        logger.info(f"Config: {args.config}")
        logger.info(f"Output: {output_dir}")
        
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
        
        # Initialize analyzer
        analyzer = ModelAnalyzer(device=args.device)
        
        # Load models
        models = {}
        
        if args.model_type in ["generator", "both"]:
            logger.info("Loading generator model...")
            generator = AetheristGenerator(config.model)
            
            if args.checkpoint:
                checkpoint = torch.load(args.checkpoint, map_location='cpu')
                if 'generator_state_dict' in checkpoint:
                    generator.load_state_dict(checkpoint['generator_state_dict'])
                else:
                    generator.load_state_dict(checkpoint)
                    
            models["generator"] = generator
            
        if args.model_type in ["discriminator", "both"]:
            logger.info("Loading discriminator model...")
            discriminator = AetheristDiscriminator(config.model)
            
            if args.checkpoint:
                checkpoint = torch.load(args.checkpoint, map_location='cpu')
                if 'discriminator_state_dict' in checkpoint:
                    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
                    
            models["discriminator"] = discriminator
            
        # Perform analysis
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"\n=== Analyzing {model_name.upper()} ===")
            
            model_results = {}
            
            # Architecture analysis
            if args.analysis_type in ["architecture", "all"]:
                logger.info("Analyzing architecture...")
                arch_analysis = analyzer.analyze_architecture(model, model_name)
                model_results["architecture"] = arch_analysis
                
                logger.info(f"Total parameters: {arch_analysis.total_parameters:,}")
                logger.info(f"Model size: {arch_analysis.model_size_mb:.2f} MB")
                logger.info(f"Layer count: {arch_analysis.layer_count}")
                
            # Performance analysis
            if args.analysis_type in ["performance", "all"]:
                logger.info("Analyzing performance...")
                
                # Determine input shape based on model type
                if model_name == "generator":
                    input_shape = (512,)  # Latent dimension
                else:
                    input_shape = (3, config.model.image_size, config.model.image_size)
                    
                perf_results = analyzer.benchmark_inference(
                    model, input_shape, model_name,
                    batch_sizes=args.batch_sizes
                )
                model_results["performance"] = perf_results
                
                # Log performance summary
                for perf in perf_results:
                    logger.info(f"Batch {perf.batch_size}: {perf.mean_inference_time:.4f}s, "
                               f"{perf.throughput_fps:.2f} samples/s, {perf.memory_usage_gb:.2f} GB")
                               
            # Quality analysis (for generator)
            if args.analysis_type in ["quality", "all"] and model_name == "generator":
                logger.info("Analyzing generation quality...")
                
                with torch.no_grad():
                    # Generate samples
                    latents = torch.randn(args.num_samples, 512)
                    if analyzer.device == "cuda":
                        latents = latents.cuda()
                        
                    # Create dummy camera matrices for generator
                    batch_size = latents.shape[0]
                    camera_matrices = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1)
                    if analyzer.device == "cuda":
                        camera_matrices = camera_matrices.cuda()
                        
                    generated_images = model(latents, camera_matrices)['high_res_image']
                    
                quality_metrics = analyzer.analyze_generation_quality(generated_images)
                model_results["quality"] = quality_metrics
                
                logger.info(f"Generated {quality_metrics.batch_size} samples at {quality_metrics.resolution}")
                
            results[model_name] = model_results
            
        # Model comparison if multiple models
        if len(models) > 1:
            logger.info("\n=== Model Comparison ===")
            comparison = analyzer.compare_models(
                models, 
                (512,) if "generator" in models else (3, config.model.image_size, config.model.image_size),
                output_dir / "model_comparison.json"
            )
            results["comparison"] = comparison
            
            # Print comparison summary
            summary = comparison.get("summary", {})
            if "recommendations" in summary:
                for rec in summary["recommendations"]:
                    logger.info(f"Recommendation: {rec}")
                    
        # Save detailed results
        logger.info("\n=== Saving Results ===")
        
        # Convert dataclass objects to dictionaries for JSON serialization
        serializable_results = {}
        for model_name, model_data in results.items():
            if model_name == "comparison":
                serializable_results[model_name] = model_data
                continue
                
            serializable_model_data = {}
            for analysis_type, data in model_data.items():
                if hasattr(data, '__dict__'):  # Dataclass object
                    serializable_model_data[analysis_type] = data.__dict__
                elif isinstance(data, list):
                    # List of dataclass objects
                    serializable_model_data[analysis_type] = [d.__dict__ if hasattr(d, '__dict__') else d for d in data]
                else:
                    serializable_model_data[analysis_type] = data
                    
            serializable_results[model_name] = serializable_model_data
            
        # Save results
        results_file = output_dir / "analysis_results.json"
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
            
        logger.info(f"Detailed results saved to {results_file}")
        
        # Generate visualizations if requested
        if args.save_visualizations:
            logger.info("Generating visualizations...")
            try:
                analyzer.visualize_results(results, output_dir)
                logger.info("Visualizations saved to output directory")
            except Exception as e:
                logger.warning(f"Failed to generate visualizations: {e}")
                
        # Generate summary report
        logger.info("\n=== Analysis Summary ===")
        
        for model_name, model_data in results.items():
            if model_name == "comparison":
                continue
                
            logger.info(f"\n{model_name.upper()}:")
            
            if "architecture" in model_data:
                arch = model_data["architecture"]
                if hasattr(arch, 'total_parameters'):
                    logger.info(f"  Parameters: {arch.total_parameters:,}")
                    logger.info(f"  Size: {arch.model_size_mb:.2f} MB")
                else:
                    logger.info(f"  Parameters: {arch['total_parameters']:,}")
                    logger.info(f"  Size: {arch['model_size_mb']:.2f} MB")
                    
            if "performance" in model_data:
                perf_list = model_data["performance"]
                if perf_list:
                    if hasattr(perf_list[0], 'throughput_fps'):
                        best_throughput = max(p.throughput_fps for p in perf_list)
                    else:
                        best_throughput = max(p['throughput_fps'] for p in perf_list)
                    logger.info(f"  Best throughput: {best_throughput:.2f} samples/s")
                    
        logger.info(f"\nAnalysis completed! Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
        
if __name__ == "__main__":
    main()
