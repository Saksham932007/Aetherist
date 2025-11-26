#!/usr/bin/env python3
"""
Aetherist Performance Benchmark Script

This script runs comprehensive performance benchmarks for Aetherist,
testing various configurations and hardware setups.

Usage:
    python scripts/benchmark_performance.py [--output results.json] [--quick]
"""

import json
import time
import torch
import psutil
import platform
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    test_name: str
    configuration: Dict[str, Any]
    inference_time: float
    memory_usage: float
    gpu_memory_usage: float
    throughput: float
    device: str
    batch_size: int
    resolution: int
    iterations: int
    timestamp: str


class PerformanceBenchmark:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self, output_file: Optional[str] = None, quick: bool = False):
        self.output_file = output_file
        self.quick = quick
        self.results = []
        
        # Determine available devices
        self.devices = ["cpu"]
        if torch.cuda.is_available():
            self.devices.extend([f"cuda:{i}" for i in range(torch.cuda.device_count())])
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.devices.append("mps")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Collect system information."""
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq().current if psutil.cpu_freq() else None,
            "memory_total": psutil.virtual_memory().total / (1024**3),  # GB
            "gpu_info": []
        }
        
        # GPU information
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info = {
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory / (1024**3),
                    "compute_capability": torch.cuda.get_device_properties(i).major
                }
                info["gpu_info"].append(gpu_info)
        
        return info
    
    def measure_memory_usage(self, device: str) -> Tuple[float, float]:
        """Measure current memory usage."""
        # System memory
        process = psutil.Process()
        system_memory = process.memory_info().rss / (1024**3)  # GB
        
        # GPU memory
        gpu_memory = 0.0
        if device.startswith("cuda"):
            gpu_id = int(device.split(":")[1]) if ":" in device else 0
            gpu_memory = torch.cuda.memory_allocated(gpu_id) / (1024**3)  # GB
        
        return system_memory, gpu_memory
    
    def create_test_model(self, config: Dict[str, Any]) -> torch.nn.Module:
        """Create a test model with given configuration."""
        # Simplified model for benchmarking
        class TestGenerator(torch.nn.Module):
            def __init__(self, latent_dim, resolution, triplane_dim):
                super().__init__()
                self.latent_dim = latent_dim
                self.resolution = resolution
                
                # Simplified architecture
                self.fc1 = torch.nn.Linear(latent_dim, triplane_dim * 4)
                self.fc2 = torch.nn.Linear(triplane_dim * 4, triplane_dim * 8)
                
                # Convolutional layers for upsampling
                self.convs = torch.nn.Sequential(
                    torch.nn.ConvTranspose2d(triplane_dim // 4, 256, 4, 2, 1),
                    torch.nn.ReLU(),
                    torch.nn.ConvTranspose2d(256, 128, 4, 2, 1),
                    torch.nn.ReLU(),
                    torch.nn.ConvTranspose2d(128, 64, 4, 2, 1),
                    torch.nn.ReLU(),
                    torch.nn.ConvTranspose2d(64, 3, 4, 2, 1),
                    torch.nn.Tanh()
                )
            
            def forward(self, z):
                x = torch.relu(self.fc1(z))
                x = torch.relu(self.fc2(x))
                x = x.view(x.size(0), -1, 4, 4)  # Reshape for conv layers
                return self.convs(x)
        
        return TestGenerator(
            config["latent_dim"],
            config["resolution"], 
            config["triplane_dim"]
        )
    
    def run_inference_benchmark(self, model: torch.nn.Module, device: str, 
                               batch_size: int, resolution: int, 
                               iterations: int = 100) -> BenchmarkResult:
        """Run inference speed benchmark."""
        model = model.to(device)
        model.eval()
        
        # Warm up
        with torch.no_grad():
            dummy_input = torch.randn(batch_size, model.latent_dim, device=device)
            for _ in range(10):
                _ = model(dummy_input)
        
        # Clear cache
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Measure memory before benchmark
        memory_before = self.measure_memory_usage(device)
        
        # Benchmark
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(iterations):
                dummy_input = torch.randn(batch_size, model.latent_dim, device=device)
                output = model(dummy_input)
                
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
        
        end_time = time.time()
        
        # Measure memory after benchmark
        memory_after = self.measure_memory_usage(device)
        
        # Calculate metrics
        total_time = end_time - start_time
        avg_inference_time = total_time / iterations
        throughput = (iterations * batch_size) / total_time  # images/second
        
        return BenchmarkResult(
            test_name="inference_speed",
            configuration={
                "latent_dim": model.latent_dim,
                "resolution": resolution,
                "triplane_dim": getattr(model, 'triplane_dim', 256)
            },
            inference_time=avg_inference_time,
            memory_usage=memory_after[0] - memory_before[0],
            gpu_memory_usage=memory_after[1] - memory_before[1],
            throughput=throughput,
            device=device,
            batch_size=batch_size,
            resolution=resolution,
            iterations=iterations,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def run_memory_scaling_benchmark(self, device: str) -> List[BenchmarkResult]:
        """Test memory usage with different batch sizes."""
        results = []
        
        if self.quick:
            batch_sizes = [1, 4]
            iterations = 20
        else:
            batch_sizes = [1, 2, 4, 8, 16, 32]
            iterations = 50
        
        base_config = {
            "latent_dim": 512,
            "resolution": 256,
            "triplane_dim": 256
        }
        
        for batch_size in batch_sizes:
            try:
                model = self.create_test_model(base_config)
                result = self.run_inference_benchmark(
                    model, device, batch_size, 256, iterations
                )
                result.test_name = "memory_scaling"
                results.append(result)
                
                print(f"✅ Batch size {batch_size}: {result.throughput:.2f} img/s, "
                      f"Memory: {result.memory_usage:.2f} GB")
                
                # Clear model
                del model
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ Batch size {batch_size}: Out of memory")
                    break
                else:
                    raise
        
        return results
    
    def run_resolution_scaling_benchmark(self, device: str) -> List[BenchmarkResult]:
        """Test performance with different resolutions."""
        results = []
        
        if self.quick:
            resolutions = [128, 256]
            iterations = 20
        else:
            resolutions = [64, 128, 256, 512]
            iterations = 50
        
        batch_size = 4
        
        for resolution in resolutions:
            try:
                config = {
                    "latent_dim": 512,
                    "resolution": resolution,
                    "triplane_dim": min(256, resolution)  # Scale triplane with resolution
                }
                
                model = self.create_test_model(config)
                result = self.run_inference_benchmark(
                    model, device, batch_size, resolution, iterations
                )
                result.test_name = "resolution_scaling"
                results.append(result)
                
                print(f"✅ Resolution {resolution}x{resolution}: {result.throughput:.2f} img/s, "
                      f"Time: {result.inference_time*1000:.1f}ms")
                
                # Clear model
                del model
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"❌ Resolution {resolution}: Out of memory")
                    break
                else:
                    raise
        
        return results
    
    def run_precision_benchmark(self, device: str) -> List[BenchmarkResult]:
        """Test different precision modes."""
        if not device.startswith("cuda"):
            return []  # Skip precision tests for non-GPU devices
        
        results = []
        precisions = ["float32", "float16"]
        if self.quick:
            iterations = 20
        else:
            iterations = 50
        
        base_config = {
            "latent_dim": 512,
            "resolution": 256,
            "triplane_dim": 256
        }
        
        for precision in precisions:
            try:
                model = self.create_test_model(base_config).to(device)
                
                if precision == "float16":
                    model = model.half()
                
                # Run benchmark
                start_time = time.time()
                
                with torch.no_grad():
                    for _ in range(iterations):
                        dummy_input = torch.randn(4, model.latent_dim, device=device)
                        if precision == "float16":
                            dummy_input = dummy_input.half()
                        
                        output = model(dummy_input)
                        torch.cuda.synchronize()
                
                end_time = time.time()
                
                avg_time = (end_time - start_time) / iterations
                throughput = (iterations * 4) / (end_time - start_time)
                
                result = BenchmarkResult(
                    test_name="precision_comparison",
                    configuration={**base_config, "precision": precision},
                    inference_time=avg_time,
                    memory_usage=0.0,  # Not measured here
                    gpu_memory_usage=torch.cuda.memory_allocated() / (1024**3),
                    throughput=throughput,
                    device=device,
                    batch_size=4,
                    resolution=256,
                    iterations=iterations,
                    timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
                )
                
                results.append(result)
                print(f"✅ {precision}: {throughput:.2f} img/s")
                
                # Clear model
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ {precision}: {str(e)}")
        
        return results
    
    def create_performance_plots(self, results: List[BenchmarkResult]) -> None:
        """Create visualization plots for benchmark results."""
        if not results:
            return
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Aetherist Performance Benchmarks', fontsize=16, fontweight='bold')
        
        # Plot 1: Throughput by batch size
        memory_results = [r for r in results if r.test_name == "memory_scaling"]
        if memory_results:
            batch_sizes = [r.batch_size for r in memory_results]
            throughputs = [r.throughput for r in memory_results]
            
            axes[0, 0].plot(batch_sizes, throughputs, 'o-', linewidth=2, markersize=8)
            axes[0, 0].set_xlabel('Batch Size')
            axes[0, 0].set_ylabel('Throughput (images/sec)')
            axes[0, 0].set_title('Throughput vs Batch Size')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].set_xscale('log', base=2)
        
        # Plot 2: Inference time by resolution
        resolution_results = [r for r in results if r.test_name == "resolution_scaling"]
        if resolution_results:
            resolutions = [r.resolution for r in resolution_results]
            times = [r.inference_time * 1000 for r in resolution_results]  # Convert to ms
            
            axes[0, 1].plot(resolutions, times, 's-', linewidth=2, markersize=8, color='orange')
            axes[0, 1].set_xlabel('Resolution')
            axes[0, 1].set_ylabel('Inference Time (ms)')
            axes[0, 1].set_title('Inference Time vs Resolution')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Memory usage by batch size
        if memory_results:
            batch_sizes = [r.batch_size for r in memory_results]
            gpu_memory = [r.gpu_memory_usage for r in memory_results]
            
            axes[1, 0].plot(batch_sizes, gpu_memory, '^-', linewidth=2, markersize=8, color='red')
            axes[1, 0].set_xlabel('Batch Size')
            axes[1, 0].set_ylabel('GPU Memory Usage (GB)')
            axes[1, 0].set_title('GPU Memory vs Batch Size')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].set_xscale('log', base=2)
        
        # Plot 4: Precision comparison
        precision_results = [r for r in results if r.test_name == "precision_comparison"]
        if precision_results:
            precisions = [r.configuration["precision"] for r in precision_results]
            throughputs = [r.throughput for r in precision_results]
            
            bars = axes[1, 1].bar(precisions, throughputs, color=['skyblue', 'lightcoral'])
            axes[1, 1].set_ylabel('Throughput (images/sec)')
            axes[1, 1].set_title('Precision Comparison')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, throughput in zip(bars, throughputs):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + throughput*0.01,
                               f'{throughput:.1f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path("benchmark_results.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Performance plots saved to: {plot_path}")
        
        plt.show()
    
    def print_summary_table(self, results: List[BenchmarkResult]) -> None:
        """Print a summary table of benchmark results."""
        if not results:
            return
        
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        
        # Group results by test type
        test_groups = {}
        for result in results:
            if result.test_name not in test_groups:
                test_groups[result.test_name] = []
            test_groups[result.test_name].append(result)
        
        for test_name, test_results in test_groups.items():
            print(f"\n📊 {test_name.upper()}:")
            print("-" * 60)
            
            if test_name == "memory_scaling":
                print(f"{'Batch Size':<12} {'Throughput':<12} {'GPU Memory':<12} {'Time (ms)':<12}")
                print("-" * 60)
                for r in test_results:
                    print(f"{r.batch_size:<12} {r.throughput:<12.2f} {r.gpu_memory_usage:<12.2f} {r.inference_time*1000:<12.1f}")
            
            elif test_name == "resolution_scaling":
                print(f"{'Resolution':<12} {'Throughput':<12} {'Time (ms)':<12} {'Memory (GB)':<12}")
                print("-" * 60)
                for r in test_results:
                    print(f"{r.resolution}x{r.resolution:<8} {r.throughput:<12.2f} {r.inference_time*1000:<12.1f} {r.gpu_memory_usage:<12.2f}")
            
            elif test_name == "precision_comparison":
                print(f"{'Precision':<12} {'Throughput':<12} {'Speedup':<12} {'Memory (GB)':<12}")
                print("-" * 60)
                base_throughput = test_results[0].throughput if test_results else 1.0
                for r in test_results:
                    speedup = r.throughput / base_throughput
                    print(f"{r.configuration['precision']:<12} {r.throughput:<12.2f} {speedup:<12.2f} {r.gpu_memory_usage:<12.2f}")
        
        # Overall statistics
        if results:
            max_throughput = max(r.throughput for r in results)
            min_time = min(r.inference_time for r in results) * 1000
            max_memory = max(r.gpu_memory_usage for r in results if r.gpu_memory_usage > 0)
            
            print(f"\n🏆 PERFORMANCE HIGHLIGHTS:")
            print(f"   Max Throughput: {max_throughput:.2f} images/sec")
            print(f"   Min Inference Time: {min_time:.1f} ms")
            if max_memory > 0:
                print(f"   Peak GPU Memory: {max_memory:.2f} GB")
    
    def run_benchmarks(self) -> None:
        """Run all benchmark suites."""
        print("🚀 Starting Aetherist Performance Benchmarks")
        print("=" * 60)
        
        # Print system information
        system_info = self.get_system_info()
        print(f"Platform: {system_info['platform']}")
        print(f"Python: {system_info['python_version']}")
        print(f"PyTorch: {system_info['pytorch_version']}")
        print(f"CPU: {system_info['cpu_count']} cores")
        if system_info['gpu_info']:
            for i, gpu in enumerate(system_info['gpu_info']):
                print(f"GPU {i}: {gpu['name']} ({gpu['memory_total']:.1f} GB)")
        print()
        
        # Run benchmarks for each device
        for device in self.devices:
            print(f"🎯 Benchmarking on device: {device}")
            print("-" * 40)
            
            try:
                # Memory scaling benchmark
                print("📈 Memory Scaling Benchmark:")
                memory_results = self.run_memory_scaling_benchmark(device)
                self.results.extend(memory_results)
                
                # Resolution scaling benchmark  
                print("\n📏 Resolution Scaling Benchmark:")
                resolution_results = self.run_resolution_scaling_benchmark(device)
                self.results.extend(resolution_results)
                
                # Precision benchmark (GPU only)
                if device.startswith("cuda"):
                    print("\n⚡ Precision Benchmark:")
                    precision_results = self.run_precision_benchmark(device)
                    self.results.extend(precision_results)
                
                print()
                
            except Exception as e:
                print(f"❌ Benchmark failed for {device}: {str(e)}")
        
        # Save results
        if self.output_file:
            results_data = {
                "system_info": system_info,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "results": [asdict(r) for r in self.results]
            }
            
            with open(self.output_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            print(f"💾 Results saved to: {self.output_file}")
        
        # Print summary
        self.print_summary_table(self.results)
        
        # Create plots
        try:
            self.create_performance_plots(self.results)
        except ImportError:
            print("⚠️ matplotlib/seaborn not available, skipping plots")
        except Exception as e:
            print(f"⚠️ Could not create plots: {e}")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description="Run Aetherist performance benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs comprehensive performance benchmarks including:
- Memory scaling with different batch sizes
- Resolution scaling tests  
- Precision comparison (FP32 vs FP16)
- Device comparison (CPU vs GPU)

Results are saved as JSON and visualized in plots.

Examples:
  python scripts/benchmark_performance.py
  python scripts/benchmark_performance.py --output results.json --quick
        """
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="benchmark_results.json",
        help="Output file for results (JSON format)"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmarks (fewer iterations and configurations)"
    )
    
    args = parser.parse_args()
    
    # Check if required packages are available
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("⚠️ matplotlib and seaborn are recommended for visualization")
        print("Install with: pip install matplotlib seaborn")
    
    # Run benchmarks
    benchmark = PerformanceBenchmark(
        output_file=args.output,
        quick=args.quick
    )
    
    try:
        benchmark.run_benchmarks()
        print("✅ Benchmarks completed successfully!")
    except KeyboardInterrupt:
        print("\n❌ Benchmarks interrupted by user")
    except Exception as e:
        print(f"❌ Benchmarks failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())