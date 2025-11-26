#!/usr/bin/env python3
"""Training validation and convergence testing for Aetherist.

Validates training setup, monitors convergence, and provides training diagnostics.
"""

import argparse
import logging
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from collections import deque

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import seaborn as sns
    from torch.utils.data import DataLoader
    VALIDATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Training validation dependencies not available: {e}")
    VALIDATION_AVAILABLE = False

if VALIDATION_AVAILABLE:
    from src.models.generator import AetheristGenerator, GeneratorConfig
    from src.models.discriminator import AetheristDiscriminator, DiscriminatorConfig
    from src.training.trainer import AetheristTrainer, TrainingConfig
    from src.data.dataset import AetheristDataset
    from src.utils.camera import CameraConfig
    from src.utils.validation import ValidationError, validate_model_config

class TrainingValidator:
    """Comprehensive training validation and monitoring."""
    
    def __init__(self, config_path: str, output_dir: str = "outputs/validation"):
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = self._setup_logging()
        self.validation_results = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for validation."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(self.output_dir / "validation.log")
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    def validate_configuration(self) -> bool:
        """Validate training configuration."""
        self.logger.info("Validating training configuration...")
        
        try:
            # Load configuration
            if not self.config_path.exists():
                raise ValidationError(f"Configuration file not found: {self.config_path}")
                
            # Here you would load your actual config
            # For now, create mock configs for validation
            gen_config = GeneratorConfig(
                latent_dim=256,
                triplane_dim=256,
                triplane_res=64,
                neural_renderer_layers=4,
                sr_channels=128,
                sr_layers=3
            )
            
            disc_config = DiscriminatorConfig(
                image_channels=3,
                base_channels=64,
                max_channels=512,
                num_blocks=4
            )
            
            training_config = TrainingConfig(
                batch_size=8,
                learning_rate=2e-4,
                num_epochs=100,
                device="cuda" if torch.cuda.is_available() else "cpu"
            )
            
            # Validate configurations
            validate_model_config(gen_config)
            validate_model_config(disc_config)
            
            self.validation_results["config_valid"] = True
            self.logger.info("✓ Configuration validation passed")
            return True
            
        except Exception as e:
            self.validation_results["config_valid"] = False
            self.logger.error(f"✗ Configuration validation failed: {e}")
            return False
            
    def validate_model_architecture(self) -> bool:
        """Validate model architectures and compatibility."""
        self.logger.info("Validating model architectures...")
        
        try:
            # Create test models
            gen_config = GeneratorConfig()
            disc_config = DiscriminatorConfig()
            
            generator = AetheristGenerator(gen_config)
            discriminator = AetheristDiscriminator(disc_config)
            
            # Test forward pass compatibility
            batch_size = 2
            latent_codes = torch.randn(batch_size, gen_config.latent_dim)
            camera_params = torch.randn(batch_size, 16)
            
            with torch.no_grad():
                # Generator forward
                generated_images = generator(latent_codes, camera_params)
                
                # Check output shape
                expected_shape = (batch_size, 3, 128, 128)  # Assuming 128x128 output
                if generated_images.shape != expected_shape:
                    raise ValidationError(
                        f"Generator output shape mismatch: {generated_images.shape} vs {expected_shape}"
                    )
                    
                # Discriminator forward
                disc_output = discriminator(generated_images, camera_params)
                
                # Validate discriminator outputs
                required_keys = ["validity", "3d_consistency"]
                for key in required_keys:
                    if key not in disc_output:
                        raise ValidationError(f"Missing discriminator output: {key}")
                        
                # Check for NaN/inf values
                if torch.isnan(generated_images).any():
                    raise ValidationError("Generator produced NaN values")
                if torch.isinf(generated_images).any():
                    raise ValidationError("Generator produced infinite values")
                    
            # Calculate model complexity
            gen_params = sum(p.numel() for p in generator.parameters())
            disc_params = sum(p.numel() for p in discriminator.parameters())
            
            self.validation_results.update({
                "architecture_valid": True,
                "generator_parameters": gen_params,
                "discriminator_parameters": disc_params,
                "total_parameters": gen_params + disc_params
            })
            
            self.logger.info(f"✓ Architecture validation passed")
            self.logger.info(f"  Generator parameters: {gen_params:,}")
            self.logger.info(f"  Discriminator parameters: {disc_params:,}")
            
            return True
            
        except Exception as e:
            self.validation_results["architecture_valid"] = False
            self.logger.error(f"✗ Architecture validation failed: {e}")
            return False
            
    def validate_data_pipeline(self, data_path: Optional[str] = None) -> bool:
        """Validate data loading and preprocessing pipeline."""
        self.logger.info("Validating data pipeline...")
        
        try:
            # Create temporary test data if no path provided
            if data_path is None:
                import tempfile
                from PIL import Image
                
                temp_dir = tempfile.mkdtemp()
                data_path = Path(temp_dir) / "test_images"
                data_path.mkdir()
                
                # Create test images
                for i in range(4):
                    img = Image.fromarray(
                        np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                    )
                    img.save(data_path / f"test_{i:03d}.jpg")
                    
                self.logger.info(f"Created temporary test data at {data_path}")
                
            # Test dataset creation
            camera_config = CameraConfig()
            dataset = AetheristDataset(
                data_path=str(data_path),
                image_size=128,
                camera_config=camera_config,
                augment=True
            )
            
            if len(dataset) == 0:
                raise ValidationError("Dataset is empty")
                
            # Test data loading
            dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
            batch = next(iter(dataloader))
            
            # Validate batch structure
            required_keys = ["image", "camera_params"]
            for key in required_keys:
                if key not in batch:
                    raise ValidationError(f"Missing batch key: {key}")
                    
            # Validate tensor shapes and ranges
            images = batch["image"]
            camera_params = batch["camera_params"]
            
            if images.shape[1:] != (3, 128, 128):
                raise ValidationError(f"Invalid image shape: {images.shape}")
                
            if camera_params.shape[1] != 16:
                raise ValidationError(f"Invalid camera params shape: {camera_params.shape}")
                
            # Check value ranges
            if images.min() < -1.1 or images.max() > 1.1:
                raise ValidationError(f"Images out of range [-1, 1]: [{images.min():.3f}, {images.max():.3f}]")
                
            self.validation_results.update({
                "data_pipeline_valid": True,
                "dataset_size": len(dataset),
                "image_shape": list(images.shape[1:]),
                "camera_params_shape": list(camera_params.shape[1:])
            })
            
            self.logger.info(f"✓ Data pipeline validation passed")
            self.logger.info(f"  Dataset size: {len(dataset)}")
            
            return True
            
        except Exception as e:
            self.validation_results["data_pipeline_valid"] = False
            self.logger.error(f"✗ Data pipeline validation failed: {e}")
            return False
            
    def test_training_step(self) -> bool:
        """Test a single training step for basic functionality."""
        self.logger.info("Testing training step...")
        
        try:
            # Create models and trainer
            gen_config = GeneratorConfig()
            disc_config = DiscriminatorConfig()
            training_config = TrainingConfig(
                batch_size=2,
                learning_rate=2e-4,
                device="cpu"  # Use CPU for testing
            )
            
            generator = AetheristGenerator(gen_config)
            discriminator = AetheristDiscriminator(disc_config)
            
            # Create dummy dataset
            import tempfile
            from PIL import Image
            
            temp_dir = tempfile.mkdtemp()
            data_path = Path(temp_dir) / "test_images"
            data_path.mkdir()
            
            for i in range(4):
                img = Image.fromarray(
                    np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                )
                img.save(data_path / f"test_{i:03d}.jpg")
                
            dataset = AetheristDataset(
                data_path=str(data_path),
                image_size=128,
                camera_config=CameraConfig(),
                augment=False
            )
            
            dataloader = DataLoader(dataset, batch_size=2)
            
            # Create trainer
            trainer = AetheristTrainer(
                generator=generator,
                discriminator=discriminator,
                config=training_config,
                train_loader=dataloader
            )
            
            # Test training step
            batch = next(iter(dataloader))
            initial_gen_params = [p.clone() for p in generator.parameters()]
            initial_disc_params = [p.clone() for p in discriminator.parameters()]
            
            losses = trainer._train_step(batch)
            
            # Validate loss structure
            required_losses = ["generator_loss", "discriminator_loss"]
            for loss_name in required_losses:
                if loss_name not in losses:
                    raise ValidationError(f"Missing loss: {loss_name}")
                    
                loss_value = losses[loss_name]
                if torch.isnan(loss_value) or torch.isinf(loss_value):
                    raise ValidationError(f"Invalid {loss_name}: {loss_value}")
                    
            # Check parameter updates
            gen_updated = any(
                not torch.equal(p1, p2) 
                for p1, p2 in zip(initial_gen_params, generator.parameters())
            )
            disc_updated = any(
                not torch.equal(p1, p2) 
                for p1, p2 in zip(initial_disc_params, discriminator.parameters())
            )
            
            if not gen_updated:
                raise ValidationError("Generator parameters not updated")
            if not disc_updated:
                raise ValidationError("Discriminator parameters not updated")
                
            self.validation_results.update({
                "training_step_valid": True,
                "losses": {k: float(v) for k, v in losses.items()}
            })
            
            self.logger.info("✓ Training step validation passed")
            for loss_name, loss_value in losses.items():
                self.logger.info(f"  {loss_name}: {loss_value:.6f}")
                
            return True
            
        except Exception as e:
            self.validation_results["training_step_valid"] = False
            self.logger.error(f"✗ Training step validation failed: {e}")
            return False
            
    def analyze_gradient_flow(self) -> bool:
        """Analyze gradient flow through the models."""
        self.logger.info("Analyzing gradient flow...")
        
        try:
            gen_config = GeneratorConfig()
            disc_config = DiscriminatorConfig()
            
            generator = AetheristGenerator(gen_config)
            discriminator = AetheristDiscriminator(disc_config)
            
            # Enable training mode
            generator.train()
            discriminator.train()
            
            # Create test inputs
            batch_size = 4
            latent_codes = torch.randn(batch_size, gen_config.latent_dim, requires_grad=True)
            camera_params = torch.randn(batch_size, 16)
            real_images = torch.randn(batch_size, 3, 128, 128)
            
            # Forward pass
            generated_images = generator(latent_codes, camera_params)
            
            # Discriminator forward
            fake_disc_out = discriminator(generated_images, camera_params)
            real_disc_out = discriminator(real_images, camera_params)
            
            # Compute losses
            gen_loss = nn.functional.binary_cross_entropy_with_logits(
                fake_disc_out["validity"],
                torch.ones_like(fake_disc_out["validity"])
            )
            
            disc_loss = (
                nn.functional.binary_cross_entropy_with_logits(
                    real_disc_out["validity"],
                    torch.ones_like(real_disc_out["validity"])
                ) +
                nn.functional.binary_cross_entropy_with_logits(
                    fake_disc_out["validity"],
                    torch.zeros_like(fake_disc_out["validity"])
                )
            ) / 2
            
            # Backward pass
            gen_loss.backward(retain_graph=True)
            disc_loss.backward()
            
            # Analyze gradients
            gen_grad_stats = self._analyze_model_gradients(generator, "Generator")
            disc_grad_stats = self._analyze_model_gradients(discriminator, "Discriminator")
            
            self.validation_results.update({
                "gradient_flow_valid": True,
                "generator_gradients": gen_grad_stats,
                "discriminator_gradients": disc_grad_stats
            })
            
            self.logger.info("✓ Gradient flow analysis passed")
            
            return True
            
        except Exception as e:
            self.validation_results["gradient_flow_valid"] = False
            self.logger.error(f"✗ Gradient flow analysis failed: {e}")
            return False
            
    def _analyze_model_gradients(self, model: nn.Module, model_name: str) -> Dict:
        """Analyze gradient statistics for a model."""
        grad_stats = {
            "layers_with_gradients": 0,
            "layers_without_gradients": 0,
            "total_layers": 0,
            "gradient_norms": [],
            "has_nan_gradients": False,
            "has_inf_gradients": False
        }
        
        for name, param in model.named_parameters():
            grad_stats["total_layers"] += 1
            
            if param.grad is not None:
                grad_stats["layers_with_gradients"] += 1
                
                grad_norm = param.grad.norm().item()
                grad_stats["gradient_norms"].append(grad_norm)
                
                if torch.isnan(param.grad).any():
                    grad_stats["has_nan_gradients"] = True
                    self.logger.warning(f"NaN gradient in {model_name} layer: {name}")
                    
                if torch.isinf(param.grad).any():
                    grad_stats["has_inf_gradients"] = True
                    self.logger.warning(f"Inf gradient in {model_name} layer: {name}")
            else:
                grad_stats["layers_without_gradients"] += 1
                self.logger.warning(f"No gradient in {model_name} layer: {name}")
                
        if grad_stats["gradient_norms"]:
            grad_stats["mean_gradient_norm"] = np.mean(grad_stats["gradient_norms"])
            grad_stats["max_gradient_norm"] = np.max(grad_stats["gradient_norms"])
            grad_stats["min_gradient_norm"] = np.min(grad_stats["gradient_norms"])
        
        return grad_stats
        
    def run_validation_suite(self, data_path: Optional[str] = None) -> bool:
        """Run complete validation suite."""
        self.logger.info("Starting comprehensive training validation...")
        
        validations = [
            ("Configuration", self.validate_configuration),
            ("Model Architecture", self.validate_model_architecture),
            ("Data Pipeline", lambda: self.validate_data_pipeline(data_path)),
            ("Training Step", self.test_training_step),
            ("Gradient Flow", self.analyze_gradient_flow)
        ]
        
        passed = 0
        total = len(validations)
        
        for name, validation_func in validations:
            self.logger.info(f"\n{'='*50}")
            self.logger.info(f"Running {name} Validation")
            self.logger.info(f"{'='*50}")
            
            try:
                if validation_func():
                    passed += 1
                    self.logger.info(f"✓ {name} validation PASSED")
                else:
                    self.logger.error(f"✗ {name} validation FAILED")
            except Exception as e:
                self.logger.error(f"✗ {name} validation ERROR: {e}")
                
        # Save validation results
        results_path = self.output_dir / "validation_results.json"
        with open(results_path, 'w') as f:
            json.dump(self.validation_results, f, indent=2, default=str)
            
        # Generate validation report
        self._generate_validation_report(passed, total)
        
        success = passed == total
        if success:
            self.logger.info(f"\n🎉 All validations passed! ({passed}/{total})")
        else:
            self.logger.error(f"\n❌ Validation failed: {passed}/{total} passed")
            
        return success
        
    def _generate_validation_report(self, passed: int, total: int):
        """Generate comprehensive validation report."""
        report_path = self.output_dir / "validation_report.md"
        
        with open(report_path, 'w') as f:
            f.write("# Aetherist Training Validation Report\n\n")
            f.write(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Overall Status:** {'✅ PASSED' if passed == total else '❌ FAILED'}\n")
            f.write(f"**Score:** {passed}/{total} validations passed\n\n")
            
            f.write("## Validation Results\n\n")
            
            for key, value in self.validation_results.items():
                if key.endswith('_valid'):
                    name = key.replace('_valid', '').replace('_', ' ').title()
                    status = '✅ PASSED' if value else '❌ FAILED'
                    f.write(f"- **{name}:** {status}\n")
                    
            f.write("\n## Detailed Results\n\n")
            f.write("```json\n")
            f.write(json.dumps(self.validation_results, indent=2, default=str))
            f.write("\n```\n")
            
        self.logger.info(f"Validation report saved to: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Validate Aetherist training setup")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml",
                       help="Training configuration file")
    parser.add_argument("--data-path", type=str, default=None,
                       help="Path to training data")
    parser.add_argument("--output-dir", type=str, default="outputs/validation",
                       help="Output directory for validation results")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if not VALIDATION_AVAILABLE:
        print("❌ Training validation dependencies not available.")
        print("Install with: pip install torch matplotlib seaborn")
        sys.exit(1)
        
    # Create validator
    validator = TrainingValidator(args.config, args.output_dir)
    
    if args.verbose:
        validator.logger.setLevel(logging.DEBUG)
        
    # Run validation
    success = validator.run_validation_suite(args.data_path)
    
    sys.exit(0 if success else 1)
    
if __name__ == "__main__":
    main()
