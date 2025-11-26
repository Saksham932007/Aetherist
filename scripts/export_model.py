#!/usr/bin/env python3
"""Model Export Utilities for Aetherist.

Supports export to ONNX, TensorRT, mobile formats, and quantized versions.
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    import torch.nn as nn
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Optional dependencies
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

try:
    import torch.backends.quantized
    QUANTIZATION_AVAILABLE = True
except ImportError:
    QUANTIZATION_AVAILABLE = False

if TORCH_AVAILABLE:
    from src.models.generator import AetheristGenerator, GeneratorConfig
    from src.models.discriminator import AetheristDiscriminator, DiscriminatorConfig
    from src.utils.performance import optimize_for_inference
    from src.utils.validation import validate_tensor_input

logger = logging.getLogger(__name__)

class ModelExporter:
    """Comprehensive model export utility."""
    
    def __init__(self, model_path: str, output_dir: str = "exports"):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for model export")
            
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.generator, self.discriminator = self._load_models()
        
        # Export metadata
        self.metadata = {
            "export_timestamp": time.time(),
            "model_path": str(self.model_path),
            "pytorch_version": torch.__version__,
            "device": str(self.device)
        }
        
        logger.info(f"Model exporter initialized")
        logger.info(f"Model path: {self.model_path}")
        logger.info(f"Output directory: {self.output_dir}")
        
    def _load_models(self) -> Tuple[nn.Module, Optional[nn.Module]]:
        """Load generator and discriminator models."""
        
        # Create model configurations
        gen_config = GeneratorConfig()
        disc_config = DiscriminatorConfig()
        
        # Initialize models
        generator = AetheristGenerator(gen_config).to(self.device)
        discriminator = AetheristDiscriminator(disc_config).to(self.device)
        
        # Load checkpoint if available
        if self.model_path.exists():
            logger.info(f"Loading checkpoint from {self.model_path}")
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            if 'generator' in checkpoint:
                generator.load_state_dict(checkpoint['generator'])
            elif 'model' in checkpoint:
                generator.load_state_dict(checkpoint['model'])
            else:
                generator.load_state_dict(checkpoint)
                
            if 'discriminator' in checkpoint:
                discriminator.load_state_dict(checkpoint['discriminator'])
            else:
                logger.warning("Discriminator weights not found in checkpoint")
                discriminator = None
        else:
            logger.warning(f"Checkpoint not found: {self.model_path}")
            logger.warning("Using randomly initialized models")
            
        # Set to evaluation mode
        generator.eval()
        if discriminator is not None:
            discriminator.eval()
            
        return generator, discriminator
        
    def export_pytorch(self, optimize: bool = True) -> str:
        """Export optimized PyTorch model."""
        logger.info("Exporting PyTorch model...")
        
        output_path = self.output_dir / "model.pth"
        
        # Optimize model if requested
        if optimize:
            optimized_generator = optimize_for_inference(self.generator)
        else:
            optimized_generator = self.generator
            
        # Save model
        torch.save({
            'generator': optimized_generator.state_dict(),
            'generator_config': optimized_generator.config.__dict__,
            'metadata': self.metadata
        }, output_path)
        
        logger.info(f"PyTorch model exported: {output_path}")
        return str(output_path)
        
    def export_torchscript(self, 
                          input_shape: Tuple[int, ...] = (1, 512),
                          camera_shape: Tuple[int, ...] = (1, 16)) -> str:
        """Export TorchScript model."""
        logger.info("Exporting TorchScript model...")
        
        output_path = self.output_dir / "model_torchscript.pt"
        
        # Create example inputs
        latent_input = torch.randn(*input_shape, device=self.device)
        camera_input = torch.randn(*camera_shape, device=self.device)
        
        try:
            # Trace the model
            with torch.no_grad():
                traced_model = torch.jit.trace(
                    self.generator, 
                    (latent_input, camera_input)
                )
                
            # Save traced model
            traced_model.save(str(output_path))
            
            # Verify the traced model
            self._verify_torchscript(traced_model, latent_input, camera_input)
            
            logger.info(f"TorchScript model exported: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"TorchScript export failed: {e}")
            raise
            
    def _verify_torchscript(self, traced_model, latent_input, camera_input):
        """Verify TorchScript model produces same output."""
        with torch.no_grad():
            original_output = self.generator(latent_input, camera_input)
            traced_output = traced_model(latent_input, camera_input)
            
            # Check output similarity
            max_diff = torch.max(torch.abs(original_output - traced_output)).item()
            logger.info(f"TorchScript verification: max difference = {max_diff:.2e}")
            
            if max_diff > 1e-4:
                logger.warning(f"Large difference detected in TorchScript model: {max_diff}")
                
    def export_onnx(self, 
                   input_shape: Tuple[int, ...] = (1, 512),
                   camera_shape: Tuple[int, ...] = (1, 16),
                   opset_version: int = 11,
                   dynamic_axes: bool = True) -> str:
        """Export ONNX model."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX not available. Install with: pip install onnx onnxruntime")
            
        logger.info("Exporting ONNX model...")
        
        output_path = self.output_dir / "model.onnx"
        
        # Create example inputs
        latent_input = torch.randn(*input_shape, device=self.device)
        camera_input = torch.randn(*camera_shape, device=self.device)
        
        # Define input names and dynamic axes
        input_names = ["latent_code", "camera_params"]
        output_names = ["generated_image"]
        
        dynamic_axes_dict = None
        if dynamic_axes:
            dynamic_axes_dict = {
                "latent_code": {0: "batch_size"},
                "camera_params": {0: "batch_size"},
                "generated_image": {0: "batch_size"}
            }
            
        try:
            # Export to ONNX
            torch.onnx.export(
                self.generator,
                (latent_input, camera_input),
                str(output_path),
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes_dict,
                opset_version=opset_version,
                do_constant_folding=True,
                verbose=False
            )
            
            # Verify ONNX model
            self._verify_onnx(str(output_path), latent_input, camera_input)
            
            logger.info(f"ONNX model exported: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"ONNX export failed: {e}")
            raise
            
    def _verify_onnx(self, onnx_path: str, latent_input: torch.Tensor, camera_input: torch.Tensor):
        """Verify ONNX model."""
        # Load and check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(onnx_path)
        
        # Convert inputs to numpy
        latent_np = latent_input.cpu().numpy()
        camera_np = camera_input.cpu().numpy()
        
        # Run ONNX inference
        ort_inputs = {
            "latent_code": latent_np,
            "camera_params": camera_np
        }
        ort_outputs = ort_session.run(None, ort_inputs)
        
        # Compare with PyTorch output
        with torch.no_grad():
            torch_output = self.generator(latent_input, camera_input)
            
        torch_np = torch_output.cpu().numpy()
        onnx_np = ort_outputs[0]
        
        max_diff = np.max(np.abs(torch_np - onnx_np))
        logger.info(f"ONNX verification: max difference = {max_diff:.2e}")
        
        if max_diff > 1e-3:
            logger.warning(f"Large difference detected in ONNX model: {max_diff}")
            
    def export_tensorrt(self, 
                       onnx_path: str,
                       max_batch_size: int = 8,
                       fp16: bool = True,
                       int8: bool = False) -> str:
        """Export TensorRT engine from ONNX model."""
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not available")
            
        logger.info("Exporting TensorRT engine...")
        
        output_path = self.output_dir / "model.trt"
        
        try:
            # Create TensorRT logger
            trt_logger = trt.Logger(trt.Logger.WARNING)
            
            # Create builder and network
            builder = trt.Builder(trt_logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, trt_logger)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as f:
                if not parser.parse(f.read()):
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    raise RuntimeError("Failed to parse ONNX model")
                    
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = 2 << 30  # 2GB
            
            if fp16 and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("Enabled FP16 optimization")
                
            if int8 and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                logger.info("Enabled INT8 optimization")
                # Note: INT8 calibration would be needed for production
                
            # Set optimization profiles for dynamic shapes
            profile = builder.create_optimization_profile()
            
            # Set dynamic input shapes
            profile.set_shape("latent_code", (1, 512), (4, 512), (max_batch_size, 512))
            profile.set_shape("camera_params", (1, 16), (4, 16), (max_batch_size, 16))
            
            config.add_optimization_profile(profile)
            
            # Build engine
            logger.info("Building TensorRT engine (this may take a while)...")
            engine = builder.build_engine(network, config)
            
            if engine is None:
                raise RuntimeError("Failed to build TensorRT engine")
                
            # Save engine
            with open(output_path, 'wb') as f:
                f.write(engine.serialize())
                
            logger.info(f"TensorRT engine exported: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"TensorRT export failed: {e}")
            raise
            
    def export_quantized(self, 
                        calibration_data: Optional[torch.Tensor] = None,
                        quantization_type: str = "dynamic") -> str:
        """Export quantized PyTorch model."""
        if not QUANTIZATION_AVAILABLE:
            raise ImportError("Quantization not available")
            
        logger.info(f"Exporting quantized model ({quantization_type})...")
        
        output_path = self.output_dir / f"model_quantized_{quantization_type}.pth"
        
        try:
            if quantization_type == "dynamic":
                # Dynamic quantization
                quantized_model = torch.quantization.quantize_dynamic(
                    self.generator, 
                    {nn.Linear, nn.Conv2d, nn.Conv3d}, 
                    dtype=torch.qint8
                )
            elif quantization_type == "static":
                # Static quantization (requires calibration)
                if calibration_data is None:
                    logger.warning("No calibration data provided, using dummy data")
                    calibration_data = torch.randn(4, 512, device=self.device)
                    
                # Prepare model for quantization
                self.generator.qconfig = torch.quantization.get_default_qconfig('fbgemm')
                quantized_model = torch.quantization.prepare(self.generator)
                
                # Calibrate with representative data
                with torch.no_grad():
                    camera_params = torch.randn(4, 16, device=self.device)
                    _ = quantized_model(calibration_data, camera_params)
                    
                # Convert to quantized model
                quantized_model = torch.quantization.convert(quantized_model)
            else:
                raise ValueError(f"Unknown quantization type: {quantization_type}")
                
            # Save quantized model
            torch.save({
                'model': quantized_model.state_dict(),
                'quantization_type': quantization_type,
                'metadata': self.metadata
            }, output_path)
            
            # Benchmark quantized model
            self._benchmark_quantized(quantized_model)
            
            logger.info(f"Quantized model exported: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Quantization export failed: {e}")
            raise
            
    def _benchmark_quantized(self, quantized_model):
        """Benchmark quantized model performance."""
        # Create test inputs
        latent_input = torch.randn(1, 512, device=self.device)
        camera_input = torch.randn(1, 16, device=self.device)
        
        # Time original model
        with torch.no_grad():
            start_time = time.time()
            for _ in range(10):
                _ = self.generator(latent_input, camera_input)
            original_time = (time.time() - start_time) / 10
            
        # Time quantized model
        with torch.no_grad():
            start_time = time.time()
            for _ in range(10):
                _ = quantized_model(latent_input, camera_input)
            quantized_time = (time.time() - start_time) / 10
            
        speedup = original_time / quantized_time
        logger.info(f"Quantization speedup: {speedup:.2f}x")
        
        # Calculate model size reduction
        original_size = sum(p.numel() * p.element_size() for p in self.generator.parameters())
        quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
        
        size_reduction = original_size / quantized_size
        logger.info(f"Model size reduction: {size_reduction:.2f}x")
        
    def export_mobile(self, 
                     input_shape: Tuple[int, ...] = (1, 512),
                     camera_shape: Tuple[int, ...] = (1, 16)) -> str:
        """Export mobile-optimized model."""
        logger.info("Exporting mobile-optimized model...")
        
        output_path = self.output_dir / "model_mobile.ptl"
        
        try:
            # Optimize model for mobile
            mobile_model = torch.jit.trace(
                self.generator,
                (torch.randn(*input_shape, device=self.device),
                 torch.randn(*camera_shape, device=self.device))
            )
            
            # Optimize for mobile
            mobile_model = torch.jit.optimize_for_inference(mobile_model)
            
            # Save mobile model
            mobile_model._save_for_lite_interpreter(str(output_path))
            
            logger.info(f"Mobile model exported: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Mobile export failed: {e}")
            raise
            
    def export_all_formats(self, 
                          include_tensorrt: bool = False,
                          include_quantized: bool = True) -> Dict[str, str]:
        """Export model to all supported formats."""
        logger.info("Exporting to all supported formats...")
        
        exports = {}
        
        try:
            # PyTorch
            exports['pytorch'] = self.export_pytorch()
            
            # TorchScript
            exports['torchscript'] = self.export_torchscript()
            
            # ONNX
            if ONNX_AVAILABLE:
                exports['onnx'] = self.export_onnx()
                
                # TensorRT (if requested and available)
                if include_tensorrt and TENSORRT_AVAILABLE:
                    exports['tensorrt'] = self.export_tensorrt(exports['onnx'])
            else:
                logger.warning("ONNX not available, skipping ONNX export")
                
            # Quantized models
            if include_quantized and QUANTIZATION_AVAILABLE:
                exports['quantized_dynamic'] = self.export_quantized(quantization_type="dynamic")
            else:
                logger.warning("Quantization not available, skipping quantized export")
                
            # Mobile
            exports['mobile'] = self.export_mobile()
            
        except Exception as e:
            logger.error(f"Error during export: {e}")
            raise
            
        # Save export manifest
        self._save_export_manifest(exports)
        
        logger.info(f"All exports completed. {len(exports)} formats exported.")
        return exports
        
    def _save_export_manifest(self, exports: Dict[str, str]):
        """Save export manifest with metadata."""
        manifest = {
            "exports": exports,
            "metadata": self.metadata,
            "model_info": {
                "generator_config": self.generator.config.__dict__,
                "total_parameters": sum(p.numel() for p in self.generator.parameters()),
                "model_size_mb": sum(p.numel() * p.element_size() for p in self.generator.parameters()) / 1024 / 1024
            }
        }
        
        manifest_path = self.output_dir / "export_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
            
        logger.info(f"Export manifest saved: {manifest_path}")

def main():
    parser = argparse.ArgumentParser(description="Export Aetherist models to various formats")
    parser.add_argument("model_path", type=str, 
                       help="Path to model checkpoint")
    parser.add_argument("--output-dir", type=str, default="exports",
                       help="Output directory for exported models")
    parser.add_argument("--format", type=str, default="all",
                       choices=["pytorch", "torchscript", "onnx", "tensorrt", 
                               "quantized", "mobile", "all"],
                       help="Export format")
    parser.add_argument("--include-tensorrt", action="store_true",
                       help="Include TensorRT export (requires ONNX)")
    parser.add_argument("--include-quantized", action="store_true", default=True,
                       help="Include quantized model export")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is required for model export")
        sys.exit(1)
        
    # Create exporter
    exporter = ModelExporter(args.model_path, args.output_dir)
    
    try:
        if args.format == "all":
            exports = exporter.export_all_formats(
                include_tensorrt=args.include_tensorrt,
                include_quantized=args.include_quantized
            )
            print("\n🎉 Export Summary:")
            for format_name, path in exports.items():
                print(f"  {format_name}: {path}")
        else:
            # Export single format
            if args.format == "pytorch":
                path = exporter.export_pytorch()
            elif args.format == "torchscript":
                path = exporter.export_torchscript()
            elif args.format == "onnx":
                path = exporter.export_onnx()
            elif args.format == "tensorrt":
                onnx_path = exporter.export_onnx()
                path = exporter.export_tensorrt(onnx_path)
            elif args.format == "quantized":
                path = exporter.export_quantized()
            elif args.format == "mobile":
                path = exporter.export_mobile()
                
            print(f"\n🎉 {args.format.title()} model exported: {path}")
            
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)
        
if __name__ == "__main__":
    main()
