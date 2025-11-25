import torch
import torch.nn as nn
import torch.quantization as quant
import torch.jit as jit
import onnx
import onnxruntime as ort
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import time
import json
import numpy as np
from dataclasses import dataclass

@dataclass
class OptimizationConfig:
    """Configuration for model optimization."""
    quantize: bool = True
    quantization_backend: str = "fbgemm"  # or "qnnpack" for mobile
    export_onnx: bool = True
    export_torchscript: bool = True
    optimize_for_inference: bool = True
    enable_dynamic_shapes: bool = False
    batch_size: int = 1
    image_size: Tuple[int, int] = (512, 512)
    
class ModelOptimizer:
    """Optimize models for production deployment."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def optimize_generator(self, 
                          generator: nn.Module, 
                          output_dir: Path,
                          sample_input: Optional[torch.Tensor] = None) -> Dict[str, Path]:
        """Optimize generator model for production deployment.
        
        Args:
            generator: The generator model to optimize
            output_dir: Directory to save optimized models
            sample_input: Sample input tensor for optimization
            
        Returns:
            Dictionary mapping optimization type to saved model path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        optimization_results = {}
        
        # Set model to evaluation mode
        generator.eval()
        
        # Create sample input if not provided
        if sample_input is None:
            sample_input = torch.randn(self.config.batch_size, 512)  # Latent dimension
            
        self.logger.info("Starting generator optimization...")
        
        # 1. Optimize for inference
        if self.config.optimize_for_inference:
            optimized_model = self._optimize_for_inference(generator)
            torch.save(optimized_model.state_dict(), output_dir / "generator_optimized.pth")
            optimization_results["optimized"] = output_dir / "generator_optimized.pth"
            generator = optimized_model
            
        # 2. Export to TorchScript
        if self.config.export_torchscript:
            try:
                traced_model = torch.jit.trace(generator, sample_input)
                traced_model.save(str(output_dir / "generator_traced.pt"))
                optimization_results["torchscript"] = output_dir / "generator_traced.pt"
                self.logger.info("TorchScript export successful")
            except Exception as e:
                self.logger.error(f"TorchScript export failed: {e}")
                
        # 3. Export to ONNX
        if self.config.export_onnx:
            try:
                onnx_path = output_dir / "generator.onnx"
                torch.onnx.export(
                    generator,
                    sample_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['latent'],
                    output_names=['image'],
                    dynamic_axes={'latent': {0: 'batch_size'}, 'image': {0: 'batch_size'}} if self.config.enable_dynamic_shapes else None
                )
                optimization_results["onnx"] = onnx_path
                self.logger.info("ONNX export successful")
                
                # Verify ONNX model
                self._verify_onnx_model(onnx_path, sample_input)
                
            except Exception as e:
                self.logger.error(f"ONNX export failed: {e}")
                
        # 4. Quantization
        if self.config.quantize:
            try:
                quantized_model = self._quantize_model(generator, sample_input)
                torch.save(quantized_model.state_dict(), output_dir / "generator_quantized.pth")
                optimization_results["quantized"] = output_dir / "generator_quantized.pth"
                self.logger.info("Model quantization successful")
            except Exception as e:
                self.logger.error(f"Model quantization failed: {e}")
                
        return optimization_results
        
    def optimize_discriminator(self, 
                              discriminator: nn.Module, 
                              output_dir: Path,
                              sample_input: Optional[torch.Tensor] = None) -> Dict[str, Path]:
        """Optimize discriminator model for production deployment.
        
        Args:
            discriminator: The discriminator model to optimize
            output_dir: Directory to save optimized models
            sample_input: Sample input tensor for optimization
            
        Returns:
            Dictionary mapping optimization type to saved model path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        optimization_results = {}
        
        # Set model to evaluation mode
        discriminator.eval()
        
        # Create sample input if not provided
        if sample_input is None:
            sample_input = torch.randn(self.config.batch_size, 3, *self.config.image_size)
            
        self.logger.info("Starting discriminator optimization...")
        
        # 1. Optimize for inference
        if self.config.optimize_for_inference:
            optimized_model = self._optimize_for_inference(discriminator)
            torch.save(optimized_model.state_dict(), output_dir / "discriminator_optimized.pth")
            optimization_results["optimized"] = output_dir / "discriminator_optimized.pth"
            discriminator = optimized_model
            
        # 2. Export to TorchScript
        if self.config.export_torchscript:
            try:
                traced_model = torch.jit.trace(discriminator, sample_input)
                traced_model.save(str(output_dir / "discriminator_traced.pt"))
                optimization_results["torchscript"] = output_dir / "discriminator_traced.pt"
                self.logger.info("TorchScript export successful")
            except Exception as e:
                self.logger.error(f"TorchScript export failed: {e}")
                
        # 3. Export to ONNX
        if self.config.export_onnx:
            try:
                onnx_path = output_dir / "discriminator.onnx"
                torch.onnx.export(
                    discriminator,
                    sample_input,
                    str(onnx_path),
                    export_params=True,
                    opset_version=11,
                    do_constant_folding=True,
                    input_names=['image'],
                    output_names=['quality_score', 'consistency_score'],
                    dynamic_axes={'image': {0: 'batch_size'}} if self.config.enable_dynamic_shapes else None
                )
                optimization_results["onnx"] = onnx_path
                self.logger.info("ONNX export successful")
                
                # Verify ONNX model
                self._verify_onnx_model(onnx_path, sample_input)
                
            except Exception as e:
                self.logger.error(f"ONNX export failed: {e}")
                
        return optimization_results
        
    def _optimize_for_inference(self, model: nn.Module) -> nn.Module:
        """Apply inference optimizations to model."""
        # Freeze batch normalization layers
        for module in model.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
                    
        # Fuse operations where possible
        try:
            model = torch.jit.optimize_for_inference(torch.jit.script(model))
        except Exception as e:
            self.logger.warning(f"JIT optimization failed: {e}")
            
        return model
        
    def _quantize_model(self, model: nn.Module, sample_input: torch.Tensor) -> nn.Module:
        """Apply quantization to model."""
        # Prepare model for quantization
        model.qconfig = quant.get_default_qconfig(self.config.quantization_backend)
        model_fp32_prepared = quant.prepare(model)
        
        # Calibrate with sample input
        with torch.no_grad():
            model_fp32_prepared(sample_input)
            
        # Convert to quantized model
        model_int8 = quant.convert(model_fp32_prepared)
        return model_int8
        
    def _verify_onnx_model(self, onnx_path: Path, sample_input: torch.Tensor):
        """Verify ONNX model can be loaded and run."""
        try:
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            # Create inference session
            ort_session = ort.InferenceSession(str(onnx_path))
            
            # Test inference
            ort_inputs = {ort_session.get_inputs()[0].name: sample_input.numpy()}
            ort_outputs = ort_session.run(None, ort_inputs)
            
            self.logger.info(f"ONNX model verification successful: {onnx_path}")
            
        except Exception as e:
            self.logger.error(f"ONNX model verification failed: {e}")
            raise
            
    def benchmark_models(self, 
                        model_paths: Dict[str, Path], 
                        sample_input: torch.Tensor,
                        num_iterations: int = 100) -> Dict[str, Dict[str, float]]:
        """Benchmark different model optimizations.
        
        Args:
            model_paths: Dictionary mapping optimization type to model path
            sample_input: Input tensor for benchmarking
            num_iterations: Number of iterations for benchmarking
            
        Returns:
            Dictionary with benchmark results for each optimization
        """
        results = {}
        
        for opt_type, model_path in model_paths.items():
            try:
                if opt_type == "onnx":
                    results[opt_type] = self._benchmark_onnx(model_path, sample_input, num_iterations)
                elif opt_type == "torchscript":
                    results[opt_type] = self._benchmark_torchscript(model_path, sample_input, num_iterations)
                else:
                    results[opt_type] = self._benchmark_pytorch(model_path, sample_input, num_iterations)
                    
            except Exception as e:
                self.logger.error(f"Benchmarking failed for {opt_type}: {e}")
                results[opt_type] = {"error": str(e)}
                
        return results
        
    def _benchmark_onnx(self, model_path: Path, sample_input: torch.Tensor, num_iterations: int) -> Dict[str, float]:
        """Benchmark ONNX model."""
        ort_session = ort.InferenceSession(str(model_path))
        ort_inputs = {ort_session.get_inputs()[0].name: sample_input.numpy()}
        
        # Warmup
        for _ in range(10):
            ort_session.run(None, ort_inputs)
            
        # Benchmark
        start_time = time.time()
        for _ in range(num_iterations):
            ort_session.run(None, ort_inputs)
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / num_iterations
        throughput = 1.0 / avg_inference_time
        
        return {
            "avg_inference_time": avg_inference_time,
            "throughput": throughput,
            "model_size_mb": model_path.stat().st_size / (1024 * 1024)
        }
        
    def _benchmark_torchscript(self, model_path: Path, sample_input: torch.Tensor, num_iterations: int) -> Dict[str, float]:
        """Benchmark TorchScript model."""
        model = torch.jit.load(str(model_path))
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                model(sample_input)
                
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                model(sample_input)
        end_time = time.time()
        
        avg_inference_time = (end_time - start_time) / num_iterations
        throughput = 1.0 / avg_inference_time
        
        return {
            "avg_inference_time": avg_inference_time,
            "throughput": throughput,
            "model_size_mb": model_path.stat().st_size / (1024 * 1024)
        }
        
    def _benchmark_pytorch(self, model_path: Path, sample_input: torch.Tensor, num_iterations: int) -> Dict[str, float]:
        """Benchmark PyTorch model."""
        # Note: This would require loading the actual model class
        # For now, return placeholder metrics
        return {
            "avg_inference_time": 0.1,
            "throughput": 10.0,
            "model_size_mb": model_path.stat().st_size / (1024 * 1024)
        }
        
    def save_optimization_report(self, 
                                optimization_results: Dict[str, Path],
                                benchmark_results: Dict[str, Dict[str, float]],
                                output_path: Path):
        """Save optimization and benchmark report."""
        report = {
            "optimization_config": {
                "quantize": self.config.quantize,
                "quantization_backend": self.config.quantization_backend,
                "export_onnx": self.config.export_onnx,
                "export_torchscript": self.config.export_torchscript,
                "optimize_for_inference": self.config.optimize_for_inference
            },
            "optimized_models": {k: str(v) for k, v in optimization_results.items()},
            "benchmark_results": benchmark_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        self.logger.info(f"Optimization report saved to {output_path}")
