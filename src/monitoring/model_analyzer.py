"""Model analysis tools for Aetherist.

Provides comprehensive analysis of model performance, quality metrics,
and architectural insights.
"""

import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.transforms import functional as TF
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

@dataclass
class ModelArchitectureAnalysis:
    """Analysis of model architecture."""
    model_name: str
    total_parameters: int
    trainable_parameters: int
    model_size_mb: float
    layer_count: int
    layer_breakdown: Dict[str, int]
    parameter_breakdown: Dict[str, int]
    flops_estimate: Optional[int] = None
    
@dataclass
class GenerationQualityMetrics:
    """Quality metrics for generated images."""
    batch_size: int
    resolution: Tuple[int, int]
    inception_score_mean: Optional[float] = None
    inception_score_std: Optional[float] = None
    fid_score: Optional[float] = None
    lpips_distance: Optional[float] = None
    ssim_score: Optional[float] = None
    psnr_score: Optional[float] = None
    
@dataclass
class InferencePerformance:
    """Inference performance metrics."""
    model_name: str
    device: str
    batch_size: int
    resolution: Tuple[int, int]
    mean_inference_time: float
    std_inference_time: float
    throughput_fps: float
    memory_usage_gb: float
    peak_memory_gb: float
    
class ModelAnalyzer:
    """Comprehensive model analysis tool."""
    
    def __init__(self, device: str = "auto"):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for model analysis")
            
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.logger = logging.getLogger(__name__)
        
    def analyze_architecture(self, model: nn.Module, model_name: str = "Unknown") -> ModelArchitectureAnalysis:
        """Analyze model architecture and parameters."""
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimate model size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / 1024**2
        
        # Layer analysis
        layer_breakdown = defaultdict(int)
        parameter_breakdown = defaultdict(int)
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                module_type = type(module).__name__
                layer_breakdown[module_type] += 1
                
                # Count parameters for this layer type
                layer_params = sum(p.numel() for p in module.parameters())
                parameter_breakdown[module_type] += layer_params
                
        return ModelArchitectureAnalysis(
            model_name=model_name,
            total_parameters=total_params,
            trainable_parameters=trainable_params,
            model_size_mb=model_size_mb,
            layer_count=len(list(model.modules())),
            layer_breakdown=dict(layer_breakdown),
            parameter_breakdown=dict(parameter_breakdown)
        )
        
    def benchmark_inference(self, 
                           model: nn.Module, 
                           input_shape: Tuple[int, ...],
                           model_name: str = "Unknown",
                           num_warmup: int = 10,
                           num_iterations: int = 100,
                           batch_sizes: Optional[List[int]] = None) -> List[InferencePerformance]:
        """Benchmark inference performance."""
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]
            
        model = model.to(self.device)
        model.eval()
        
        results = []
        
        for batch_size in batch_sizes:
            self.logger.info(f"Benchmarking batch size {batch_size}...")
            
            # Create input tensor
            if len(input_shape) == 1:  # Latent input
                input_tensor = torch.randn(batch_size, *input_shape, device=self.device)
            else:  # Image input
                input_tensor = torch.randn(batch_size, *input_shape, device=self.device)
                
            # Warmup
            with torch.no_grad():
                for _ in range(num_warmup):
                    try:
                        _ = model(input_tensor)
                    except Exception as e:
                        self.logger.warning(f"Warmup failed for batch size {batch_size}: {e}")
                        break
                        
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                
            # Benchmark
            inference_times = []
            
            with torch.no_grad():
                for _ in range(num_iterations):
                    start_time = time.perf_counter()
                    
                    try:
                        _ = model(input_tensor)
                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                    except Exception as e:
                        self.logger.warning(f"Inference failed: {e}")
                        continue
                        
                    end_time = time.perf_counter()
                    inference_times.append(end_time - start_time)
                    
            if not inference_times:
                self.logger.warning(f"No successful inferences for batch size {batch_size}")
                continue
                
            # Calculate metrics
            mean_time = np.mean(inference_times)
            std_time = np.std(inference_times)
            throughput = batch_size / mean_time
            
            # Memory usage
            memory_usage = 0
            peak_memory = 0
            if torch.cuda.is_available():
                memory_usage = torch.cuda.memory_allocated(self.device) / 1024**3
                peak_memory = torch.cuda.max_memory_allocated(self.device) / 1024**3
                
            performance = InferencePerformance(
                model_name=model_name,
                device=self.device,
                batch_size=batch_size,
                resolution=input_shape[-2:] if len(input_shape) > 1 else (0, 0),
                mean_inference_time=mean_time,
                std_inference_time=std_time,
                throughput_fps=throughput,
                memory_usage_gb=memory_usage,
                peak_memory_gb=peak_memory
            )
            
            results.append(performance)
            
        return results
        
    def analyze_generation_quality(self, 
                                  generated_images: torch.Tensor,
                                  real_images: Optional[torch.Tensor] = None) -> GenerationQualityMetrics:
        """Analyze quality of generated images."""
        batch_size = generated_images.shape[0]
        resolution = (generated_images.shape[-2], generated_images.shape[-1])
        
        metrics = GenerationQualityMetrics(
            batch_size=batch_size,
            resolution=resolution
        )
        
        # Convert to numpy for analysis
        if isinstance(generated_images, torch.Tensor):
            gen_images_np = generated_images.detach().cpu().numpy()
        else:
            gen_images_np = generated_images
            
        # Basic image statistics
        gen_mean = np.mean(gen_images_np)
        gen_std = np.std(gen_images_np)
        
        self.logger.info(f"Generated images - Mean: {gen_mean:.4f}, Std: {gen_std:.4f}")
        
        # If real images provided, compute comparison metrics
        if real_images is not None:
            try:
                # SSIM calculation (simplified)
                ssim_scores = []
                for i in range(min(batch_size, real_images.shape[0])):
                    ssim = self._compute_ssim(generated_images[i], real_images[i])
                    ssim_scores.append(ssim)
                    
                if ssim_scores:
                    metrics.ssim_score = float(np.mean(ssim_scores))
                    
            except Exception as e:
                self.logger.warning(f"SSIM computation failed: {e}")
                
        return metrics
        
    def _compute_ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> float:
        """Compute SSIM between two images (simplified version)."""
        # Convert to grayscale if needed
        if img1.dim() == 3 and img1.shape[0] == 3:
            img1 = torch.mean(img1, dim=0, keepdim=True)
        if img2.dim() == 3 and img2.shape[0] == 3:
            img2 = torch.mean(img2, dim=0, keepdim=True)
            
        # Normalize to [0, 1]
        img1 = (img1 + 1) / 2  # Assuming [-1, 1] range
        img2 = (img2 + 1) / 2
        
        # SSIM constants
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        
        # Convert to [0, 255] range
        img1 = img1 * 255
        img2 = img2 * 255
        
        mu1 = torch.mean(img1)
        mu2 = torch.mean(img2)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = torch.var(img1)
        sigma2_sq = torch.var(img2)
        sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))
        
        ssim = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return float(ssim)
        
    def compare_models(self, 
                      models: Dict[str, nn.Module],
                      input_shape: Tuple[int, ...],
                      save_path: Optional[Path] = None) -> Dict[str, Any]:
        """Compare multiple models."""
        comparison = {
            "timestamp": time.time(),
            "device": self.device,
            "input_shape": input_shape,
            "models": {}
        }
        
        for model_name, model in models.items():
            self.logger.info(f"Analyzing {model_name}...")
            
            # Architecture analysis
            arch_analysis = self.analyze_architecture(model, model_name)
            
            # Performance benchmark
            perf_results = self.benchmark_inference(
                model, input_shape, model_name, 
                num_warmup=5, num_iterations=50,
                batch_sizes=[1, 2, 4]
            )
            
            comparison["models"][model_name] = {
                "architecture": asdict(arch_analysis),
                "performance": [asdict(p) for p in perf_results]
            }
            
        # Create summary comparison
        summary = self._create_comparison_summary(comparison)
        comparison["summary"] = summary
        
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(comparison, f, indent=2)
            self.logger.info(f"Model comparison saved to {save_path}")
            
        return comparison
        
    def _create_comparison_summary(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of model comparison."""
        models_data = comparison["models"]
        
        summary = {
            "parameter_comparison": {},
            "performance_comparison": {},
            "recommendations": []
        }
        
        # Parameter comparison
        for model_name, data in models_data.items():
            arch = data["architecture"]
            summary["parameter_comparison"][model_name] = {
                "total_parameters": arch["total_parameters"],
                "model_size_mb": arch["model_size_mb"],
                "parameter_efficiency": arch["total_parameters"] / arch["model_size_mb"]
            }
            
        # Performance comparison (batch size 1)
        for model_name, data in models_data.items():
            perf = data["performance"]
            batch_1_perf = next((p for p in perf if p["batch_size"] == 1), None)
            if batch_1_perf:
                summary["performance_comparison"][model_name] = {
                    "inference_time": batch_1_perf["mean_inference_time"],
                    "throughput_fps": batch_1_perf["throughput_fps"],
                    "memory_usage_gb": batch_1_perf["memory_usage_gb"]
                }
                
        # Generate recommendations
        param_data = summary["parameter_comparison"]
        perf_data = summary["performance_comparison"]
        
        if param_data and perf_data:
            # Find most efficient model
            fastest_model = min(perf_data.items(), key=lambda x: x[1]["inference_time"])[0]
            smallest_model = min(param_data.items(), key=lambda x: x[1]["model_size_mb"])[0]
            
            summary["recommendations"].append(f"Fastest inference: {fastest_model}")
            summary["recommendations"].append(f"Smallest model: {smallest_model}")
            
        return summary
        
    def visualize_results(self, 
                         analysis_results: Dict[str, Any], 
                         save_dir: Optional[Path] = None) -> None:
        """Visualize analysis results."""
        if not PLOTTING_AVAILABLE:
            self.logger.warning("Matplotlib not available for visualization")
            return
            
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
        # Model parameter comparison
        self._plot_parameter_comparison(analysis_results, save_dir)
        
        # Performance comparison
        self._plot_performance_comparison(analysis_results, save_dir)
        
    def _plot_parameter_comparison(self, results: Dict[str, Any], save_dir: Optional[Path] = None):
        """Plot model parameter comparison."""
        models_data = results.get("models", {})
        if not models_data:
            return
            
        model_names = list(models_data.keys())
        total_params = [data["architecture"]["total_parameters"] for data in models_data.values()]
        model_sizes = [data["architecture"]["model_size_mb"] for data in models_data.values()]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Parameter count
        ax1.bar(model_names, total_params)
        ax1.set_title("Total Parameters")
        ax1.set_ylabel("Parameters (count)")
        ax1.tick_params(axis='x', rotation=45)
        
        # Model size
        ax2.bar(model_names, model_sizes)
        ax2.set_title("Model Size")
        ax2.set_ylabel("Size (MB)")
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(save_dir / "parameter_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
        
    def _plot_performance_comparison(self, results: Dict[str, Any], save_dir: Optional[Path] = None):
        """Plot performance comparison."""
        models_data = results.get("models", {})
        if not models_data:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        for model_name, data in models_data.items():
            perf = data["performance"]
            batch_sizes = [p["batch_size"] for p in perf]
            inference_times = [p["mean_inference_time"] for p in perf]
            throughputs = [p["throughput_fps"] for p in perf]
            
            ax1.plot(batch_sizes, inference_times, marker='o', label=model_name)
            ax2.plot(batch_sizes, throughputs, marker='o', label=model_name)
            
        ax1.set_xlabel("Batch Size")
        ax1.set_ylabel("Inference Time (s)")
        ax1.set_title("Inference Time vs Batch Size")
        ax1.legend()
        ax1.grid(True)
        
        ax2.set_xlabel("Batch Size")
        ax2.set_ylabel("Throughput (samples/s)")
        ax2.set_title("Throughput vs Batch Size")
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(save_dir / "performance_comparison.png", dpi=150, bbox_inches='tight')
        plt.show()
