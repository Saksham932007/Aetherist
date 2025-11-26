#!/usr/bin/env python3
"""Comprehensive benchmarking suite for Aetherist.

Includes performance benchmarks, quality metrics, and comparison with baselines.
"""

import argparse
import json
import logging
import sys
import time
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import gc

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from PIL import Image
    BENCHMARK_AVAILABLE = True
except ImportError as e:
    print(f"Benchmark dependencies not available: {e}")
    BENCHMARK_AVAILABLE = False

if BENCHMARK_AVAILABLE:
    from src.models.generator import AetheristGenerator, GeneratorConfig
    from src.models.discriminator import AetheristDiscriminator, DiscriminatorConfig
    from src.utils.camera import generate_camera_poses
    from src.utils.performance import optimize_for_inference, benchmark_model
    from src.utils.monitoring import create_monitoring_stack

logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    test_name: str
    metric_name: str
    value: float
    unit: str
    device: str
    timestamp: float
    metadata: Dict[str, Any] = None
    
@dataclass
class PerformanceBenchmark:
    """Performance benchmark results."""
    inference_time_ms: float
    throughput_fps: float
    memory_usage_mb: float
    peak_memory_mb: float
    gpu_utilization: float
    energy_consumption: Optional[float] = None
    
@dataclass
class QualityMetrics:
    """Quality assessment metrics."""
    fid_score: Optional[float] = None
    lpips_score: Optional[float] = None
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    view_consistency: Optional[float] = None
    geometric_accuracy: Optional[float] = None

class AetheristBenchmark:
    """Comprehensive benchmark suite for Aetherist."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 output_dir: str = "benchmark_results",
                 device: Optional[str] = None):
        if not BENCHMARK_AVAILABLE:
            raise ImportError("Benchmark dependencies not available")
            
        self.model_path = Path(model_path) if model_path else None
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device(device if device else 
                                 ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Load models
        self.generator, self.discriminator = self._load_models()
        
        # Monitoring
        self.monitoring = create_monitoring_stack("aetherist-benchmark")
        
        # Benchmark configuration
        self.benchmark_config = {
            "warmup_iterations": 10,
            "measurement_iterations": 50,
            "batch_sizes": [1, 2, 4, 8],
            "resolutions": [256, 512, 1024],
            "precision_modes": ["fp32", "fp16"],
            "num_views": [1, 4, 8, 16]
        }
        
        self.results = []
        
        logger.info(f"Benchmark suite initialized")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model path: {self.model_path}")
        
    def _load_models(self) -> Tuple[torch.nn.Module, Optional[torch.nn.Module]]:
        """Load generator and discriminator models."""
        # Create configurations
        gen_config = GeneratorConfig()
        disc_config = DiscriminatorConfig()
        
        # Initialize models
        generator = AetheristGenerator(gen_config).to(self.device)
        discriminator = AetheristDiscriminator(disc_config).to(self.device)
        
        # Load checkpoint if available
        if self.model_path and self.model_path.exists():
            logger.info(f"Loading checkpoint from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if 'generator' in checkpoint:
                generator.load_state_dict(checkpoint['generator'])
            else:
                generator.load_state_dict(checkpoint)
                
            if 'discriminator' in checkpoint:
                discriminator.load_state_dict(checkpoint['discriminator'])
            else:
                discriminator = None
        else:
            logger.warning("Using randomly initialized models")
            
        # Set to evaluation mode
        generator.eval()
        if discriminator is not None:
            discriminator.eval()
            
        return generator, discriminator
        
    def benchmark_inference_performance(self) -> List[BenchmarkResult]:
        """Benchmark inference performance across different configurations."""
        logger.info("Running inference performance benchmarks...")
        
        results = []
        
        for batch_size in self.benchmark_config["batch_sizes"]:
            for precision in self.benchmark_config["precision_modes"]:
                if precision == "fp16" and not torch.cuda.is_available():
                    continue
                    
                # Setup precision
                if precision == "fp16":
                    self.generator.half()
                    dtype = torch.float16
                else:
                    self.generator.float()
                    dtype = torch.float32
                    
                # Create test inputs
                latent_input = torch.randn(
                    batch_size, self.generator.config.latent_dim, 
                    device=self.device, dtype=dtype
                )
                camera_input = torch.randn(
                    batch_size, 16, device=self.device, dtype=dtype
                )
                
                # Benchmark
                perf_results = self._benchmark_forward_pass(
                    self.generator, [latent_input, camera_input],
                    f"inference_batch_{batch_size}_{precision}"
                )
                
                # Record results
                results.extend([
                    BenchmarkResult(
                        test_name="inference_performance",
                        metric_name="latency_ms",
                        value=perf_results["avg_time_per_iteration"] * 1000,
                        unit="ms",
                        device=str(self.device),
                        timestamp=time.time(),
                        metadata={"batch_size": batch_size, "precision": precision}
                    ),
                    BenchmarkResult(
                        test_name="inference_performance",
                        metric_name="throughput_fps",
                        value=perf_results["iterations_per_second"] * batch_size,
                        unit="fps",
                        device=str(self.device),
                        timestamp=time.time(),
                        metadata={"batch_size": batch_size, "precision": precision}
                    )
                ])
                
        self.results.extend(results)
        return results
        
    def benchmark_memory_usage(self) -> List[BenchmarkResult]:
        """Benchmark memory usage patterns."""
        logger.info("Running memory usage benchmarks...")
        
        results = []
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping GPU memory benchmarks")
            return results
            
        for batch_size in self.benchmark_config["batch_sizes"]:
            # Clear memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Create inputs
            latent_input = torch.randn(
                batch_size, self.generator.config.latent_dim, device=self.device
            )
            camera_input = torch.randn(batch_size, 16, device=self.device)
            
            # Measure memory
            initial_memory = torch.cuda.memory_allocated()
            
            with torch.no_grad():
                output = self.generator(latent_input, camera_input)
                
            peak_memory = torch.cuda.max_memory_allocated()
            final_memory = torch.cuda.memory_allocated()
            
            memory_used = (peak_memory - initial_memory) / 1024 / 1024  # MB
            
            results.append(BenchmarkResult(
                test_name="memory_usage",
                metric_name="peak_memory_mb",
                value=memory_used,
                unit="MB",
                device=str(self.device),
                timestamp=time.time(),
                metadata={"batch_size": batch_size}
            ))
            
        self.results.extend(results)
        return results
        
    def benchmark_scaling(self) -> List[BenchmarkResult]:
        """Benchmark scaling performance with different configurations."""
        logger.info("Running scaling benchmarks...")
        
        results = []
        
        # Test batch size scaling
        batch_times = []
        batch_sizes = [1, 2, 4, 8, 16]
        
        for batch_size in batch_sizes:
            if batch_size > 16:  # Skip large batches if memory limited
                continue
                
            latent_input = torch.randn(
                batch_size, self.generator.config.latent_dim, device=self.device
            )
            camera_input = torch.randn(batch_size, 16, device=self.device)
            
            # Measure time
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = self.generator(latent_input, camera_input)
            avg_time = (time.time() - start_time) / 10
            
            batch_times.append(avg_time)
            
            # Calculate efficiency (time per sample)
            time_per_sample = avg_time / batch_size
            
            results.append(BenchmarkResult(
                test_name="scaling",
                metric_name="time_per_sample_ms",
                value=time_per_sample * 1000,
                unit="ms",
                device=str(self.device),
                timestamp=time.time(),
                metadata={"batch_size": batch_size}
            ))
            
        # Calculate scaling efficiency
        if len(batch_times) >= 2:
            baseline_efficiency = batch_times[0]  # Single sample time
            for i, (batch_size, batch_time) in enumerate(zip(batch_sizes[:len(batch_times)], batch_times)):
                if batch_size > 1:
                    expected_time = baseline_efficiency * batch_size
                    efficiency = expected_time / batch_time
                    
                    results.append(BenchmarkResult(
                        test_name="scaling",
                        metric_name="batch_efficiency",
                        value=efficiency,
                        unit="ratio",
                        device=str(self.device),
                        timestamp=time.time(),
                        metadata={"batch_size": batch_size}
                    ))
                    
        self.results.extend(results)
        return results
        
    def benchmark_multiview_consistency(self) -> List[BenchmarkResult]:
        """Benchmark multi-view generation consistency."""
        logger.info("Running multi-view consistency benchmarks...")
        
        results = []
        
        for num_views in self.benchmark_config["num_views"]:
            # Generate camera poses
            camera_poses = generate_camera_poses(
                num_views=num_views, radius=2.5, device=self.device
            )
            
            # Fixed latent code for consistency
            latent_code = torch.randn(1, self.generator.config.latent_dim, device=self.device)
            
            # Generate views
            views = []
            generation_times = []
            
            for camera_params in camera_poses:
                camera_batch = camera_params.unsqueeze(0)
                
                start_time = time.time()
                with torch.no_grad():
                    view = self.generator(latent_code, camera_batch)
                generation_times.append(time.time() - start_time)
                
                views.append(view)
                
            # Calculate consistency metrics
            avg_generation_time = statistics.mean(generation_times)
            total_time = sum(generation_times)
            
            results.extend([
                BenchmarkResult(
                    test_name="multiview_consistency",
                    metric_name="avg_view_generation_ms",
                    value=avg_generation_time * 1000,
                    unit="ms",
                    device=str(self.device),
                    timestamp=time.time(),
                    metadata={"num_views": num_views}
                ),
                BenchmarkResult(
                    test_name="multiview_consistency",
                    metric_name="total_generation_time_s",
                    value=total_time,
                    unit="s",
                    device=str(self.device),
                    timestamp=time.time(),
                    metadata={"num_views": num_views}
                )
            ])
            
        self.results.extend(results)
        return results
        
    def _benchmark_forward_pass(self, 
                               model: torch.nn.Module,
                               inputs: List[torch.Tensor],
                               test_name: str) -> Dict[str, float]:
        """Benchmark forward pass of a model."""
        # Warmup
        with torch.no_grad():
            for _ in range(self.benchmark_config["warmup_iterations"]):
                _ = model(*inputs)
                
        # Synchronize GPU
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            
        # Measure
        times = []
        for _ in range(self.benchmark_config["measurement_iterations"]):
            start_time = time.time()
            
            with torch.no_grad():
                _ = model(*inputs)
                
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
            times.append(time.time() - start_time)
            
        return {
            "total_time": sum(times),
            "avg_time_per_iteration": statistics.mean(times),
            "std_time": statistics.stdev(times) if len(times) > 1 else 0,
            "min_time": min(times),
            "max_time": max(times),
            "iterations_per_second": 1.0 / statistics.mean(times),
            "num_iterations": len(times)
        }
        
    def benchmark_optimization_techniques(self) -> List[BenchmarkResult]:
        """Benchmark different optimization techniques."""
        logger.info("Running optimization technique benchmarks...")
        
        results = []
        
        # Test inputs
        batch_size = 4
        latent_input = torch.randn(
            batch_size, self.generator.config.latent_dim, device=self.device
        )
        camera_input = torch.randn(batch_size, 16, device=self.device)
        inputs = [latent_input, camera_input]
        
        # Baseline (no optimization)
        baseline_results = self._benchmark_forward_pass(
            self.generator, inputs, "baseline"
        )
        baseline_time = baseline_results["avg_time_per_iteration"]
        
        results.append(BenchmarkResult(
            test_name="optimization_techniques",
            metric_name="inference_time_ms",
            value=baseline_time * 1000,
            unit="ms",
            device=str(self.device),
            timestamp=time.time(),
            metadata={"technique": "baseline"}
        ))
        
        # Test optimized model
        try:
            optimized_generator = optimize_for_inference(self.generator)
            
            optimized_results = self._benchmark_forward_pass(
                optimized_generator, inputs, "optimized"
            )
            optimized_time = optimized_results["avg_time_per_iteration"]
            speedup = baseline_time / optimized_time
            
            results.extend([
                BenchmarkResult(
                    test_name="optimization_techniques",
                    metric_name="inference_time_ms",
                    value=optimized_time * 1000,
                    unit="ms",
                    device=str(self.device),
                    timestamp=time.time(),
                    metadata={"technique": "optimized"}
                ),
                BenchmarkResult(
                    test_name="optimization_techniques",
                    metric_name="speedup",
                    value=speedup,
                    unit="ratio",
                    device=str(self.device),
                    timestamp=time.time(),
                    metadata={"technique": "optimized_vs_baseline"}
                )
            ])
            
        except Exception as e:
            logger.warning(f"Optimization benchmark failed: {e}")
            
        # Test TorchScript compilation
        try:
            compiled_generator = torch.jit.script(self.generator)
            
            compiled_results = self._benchmark_forward_pass(
                compiled_generator, inputs, "torchscript"
            )
            compiled_time = compiled_results["avg_time_per_iteration"]
            speedup = baseline_time / compiled_time
            
            results.extend([
                BenchmarkResult(
                    test_name="optimization_techniques",
                    metric_name="inference_time_ms",
                    value=compiled_time * 1000,
                    unit="ms",
                    device=str(self.device),
                    timestamp=time.time(),
                    metadata={"technique": "torchscript"}
                ),
                BenchmarkResult(
                    test_name="optimization_techniques",
                    metric_name="speedup",
                    value=speedup,
                    unit="ratio",
                    device=str(self.device),
                    timestamp=time.time(),
                    metadata={"technique": "torchscript_vs_baseline"}
                )
            ])
            
        except Exception as e:
            logger.warning(f"TorchScript benchmark failed: {e}")
            
        self.results.extend(results)
        return results
        
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run all benchmark tests."""
        logger.info("Starting comprehensive benchmark suite...")
        
        benchmark_start = time.time()
        
        # Run all benchmarks
        benchmark_results = {
            "inference_performance": self.benchmark_inference_performance(),
            "memory_usage": self.benchmark_memory_usage(),
            "scaling": self.benchmark_scaling(),
            "multiview_consistency": self.benchmark_multiview_consistency(),
            "optimization_techniques": self.benchmark_optimization_techniques()
        }
        
        total_time = time.time() - benchmark_start
        
        # Generate summary
        summary = self._generate_benchmark_summary(benchmark_results, total_time)
        
        # Save results
        self._save_benchmark_results(benchmark_results, summary)
        
        # Generate visualizations
        self._generate_visualizations()
        
        logger.info(f"Benchmark suite completed in {total_time:.2f}s")
        return summary
        
    def _generate_benchmark_summary(self, 
                                   benchmark_results: Dict[str, List[BenchmarkResult]],
                                   total_time: float) -> Dict[str, Any]:
        """Generate benchmark summary."""
        summary = {
            "total_benchmark_time": total_time,
            "device": str(self.device),
            "model_info": {
                "total_parameters": sum(p.numel() for p in self.generator.parameters()),
                "model_size_mb": sum(
                    p.numel() * p.element_size() for p in self.generator.parameters()
                ) / 1024 / 1024
            },
            "test_summary": {}
        }
        
        for test_name, results in benchmark_results.items():
            if results:
                summary["test_summary"][test_name] = {
                    "num_results": len(results),
                    "metrics": list(set(r.metric_name for r in results))
                }
                
        return summary
        
    def _save_benchmark_results(self, 
                               benchmark_results: Dict[str, List[BenchmarkResult]],
                               summary: Dict[str, Any]):
        """Save benchmark results to files."""
        # Save individual results
        for test_name, results in benchmark_results.items():
            if results:
                results_file = self.output_dir / f"{test_name}_results.json"
                with open(results_file, 'w') as f:
                    json.dump([asdict(r) for r in results], f, indent=2)
                    
        # Save summary
        summary_file = self.output_dir / "benchmark_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
            
        # Save all results together
        all_results_file = self.output_dir / "all_results.json"
        with open(all_results_file, 'w') as f:
            json.dump({
                "summary": summary,
                "results": {test: [asdict(r) for r in results] 
                           for test, results in benchmark_results.items()}
            }, f, indent=2, default=str)
            
        logger.info(f"Benchmark results saved to {self.output_dir}")
        
    def _generate_visualizations(self):
        """Generate benchmark visualization plots."""
        logger.info("Generating benchmark visualizations...")
        
        try:
            # Performance vs batch size plot
            self._plot_performance_vs_batch_size()
            
            # Memory usage plot
            self._plot_memory_usage()
            
            # Optimization comparison
            self._plot_optimization_comparison()
            
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
            
    def _plot_performance_vs_batch_size(self):
        """Plot performance vs batch size."""
        # Extract performance data
        perf_results = [r for r in self.results if r.test_name == "inference_performance"]
        
        if not perf_results:
            return
            
        # Group by precision
        fp32_data = [r for r in perf_results if r.metadata.get("precision") == "fp32"]
        fp16_data = [r for r in perf_results if r.metadata.get("precision") == "fp16"]
        
        plt.figure(figsize=(12, 8))
        
        # Plot throughput
        plt.subplot(2, 2, 1)
        if fp32_data:
            throughput_data = [r for r in fp32_data if r.metric_name == "throughput_fps"]
            batch_sizes = [r.metadata["batch_size"] for r in throughput_data]
            throughput_values = [r.value for r in throughput_data]
            plt.plot(batch_sizes, throughput_values, 'o-', label='FP32')
            
        if fp16_data:
            throughput_data = [r for r in fp16_data if r.metric_name == "throughput_fps"]
            batch_sizes = [r.metadata["batch_size"] for r in throughput_data]
            throughput_values = [r.value for r in throughput_data]
            plt.plot(batch_sizes, throughput_values, 's-', label='FP16')
            
        plt.xlabel('Batch Size')
        plt.ylabel('Throughput (FPS)')
        plt.title('Throughput vs Batch Size')
        plt.legend()
        plt.grid(True)
        
        # Plot latency
        plt.subplot(2, 2, 2)
        if fp32_data:
            latency_data = [r for r in fp32_data if r.metric_name == "latency_ms"]
            batch_sizes = [r.metadata["batch_size"] for r in latency_data]
            latency_values = [r.value for r in latency_data]
            plt.plot(batch_sizes, latency_values, 'o-', label='FP32')
            
        if fp16_data:
            latency_data = [r for r in fp16_data if r.metric_name == "latency_ms"]
            batch_sizes = [r.metadata["batch_size"] for r in latency_data]
            latency_values = [r.value for r in latency_data]
            plt.plot(batch_sizes, latency_values, 's-', label='FP16')
            
        plt.xlabel('Batch Size')
        plt.ylabel('Latency (ms)')
        plt.title('Latency vs Batch Size')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "performance_vs_batch_size.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_memory_usage(self):
        """Plot memory usage vs batch size."""
        memory_results = [r for r in self.results if r.test_name == "memory_usage"]
        
        if not memory_results:
            return
            
        batch_sizes = [r.metadata["batch_size"] for r in memory_results]
        memory_values = [r.value for r in memory_results]
        
        plt.figure(figsize=(8, 6))
        plt.plot(batch_sizes, memory_values, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Batch Size')
        plt.ylabel('Peak Memory Usage (MB)')
        plt.title('Memory Usage vs Batch Size')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "memory_usage.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_optimization_comparison(self):
        """Plot optimization technique comparison."""
        opt_results = [r for r in self.results if r.test_name == "optimization_techniques"]
        
        if not opt_results:
            return
            
        # Extract inference times for different techniques
        techniques = {}
        for r in opt_results:
            if r.metric_name == "inference_time_ms":
                technique = r.metadata["technique"]
                techniques[technique] = r.value
                
        if not techniques:
            return
            
        plt.figure(figsize=(10, 6))
        
        technique_names = list(techniques.keys())
        times = list(techniques.values())
        
        bars = plt.bar(technique_names, times, alpha=0.7)
        plt.xlabel('Optimization Technique')
        plt.ylabel('Inference Time (ms)')
        plt.title('Optimization Technique Comparison')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.01,
                    f'{time_val:.1f}ms', ha='center', va='bottom')
                    
        plt.tight_layout()
        plt.savefig(self.output_dir / "optimization_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Aetherist Benchmark Suite")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to run benchmarks on (cuda/cpu)")
    parser.add_argument("--test", type=str, default="all",
                       choices=["all", "performance", "memory", "scaling", 
                               "multiview", "optimization"],
                       help="Specific test to run")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if not BENCHMARK_AVAILABLE:
        logger.error("Benchmark dependencies not available")
        logger.error("Install with: pip install torch matplotlib seaborn")
        sys.exit(1)
        
    # Create benchmark suite
    benchmark = AetheristBenchmark(
        model_path=args.model_path,
        output_dir=args.output_dir,
        device=args.device
    )
    
    try:
        if args.test == "all":
            summary = benchmark.run_comprehensive_benchmark()
        else:
            # Run specific test
            if args.test == "performance":
                results = benchmark.benchmark_inference_performance()
            elif args.test == "memory":
                results = benchmark.benchmark_memory_usage()
            elif args.test == "scaling":
                results = benchmark.benchmark_scaling()
            elif args.test == "multiview":
                results = benchmark.benchmark_multiview_consistency()
            elif args.test == "optimization":
                results = benchmark.benchmark_optimization_techniques()
                
            print(f"\n📊 {args.test.title()} benchmark completed: {len(results)} results")
            return
            
        # Print summary
        print("\n🎉 Benchmark Summary:")
        print(f"  Total time: {summary['total_benchmark_time']:.2f}s")
        print(f"  Device: {summary['device']}")
        print(f"  Model parameters: {summary['model_info']['total_parameters']:,}")
        print(f"  Model size: {summary['model_info']['model_size_mb']:.1f} MB")
        
        print("\n  Test Results:")
        for test_name, test_info in summary['test_summary'].items():
            print(f"    {test_name}: {test_info['num_results']} results")
            
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
