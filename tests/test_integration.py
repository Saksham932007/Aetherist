#!/usr/bin/env python3
"""Integration tests for Aetherist system.

Tests the complete pipeline from data loading to generation.
"""

import pytest
import torch
import tempfile
import json
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.generator import AetheristGenerator, GeneratorConfig
from src.models.discriminator import AetheristDiscriminator, DiscriminatorConfig
from src.training.trainer import AetheristTrainer, TrainingConfig
from src.inference.pipeline import InferencePipeline
from src.data.dataset import AetheristDataset
from src.utils.camera import CameraConfig

class TestE2EIntegration:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def generator_config(self):
        """Create test generator configuration."""
        return GeneratorConfig(
            latent_dim=64,
            triplane_dim=32,
            triplane_res=32,
            neural_renderer_layers=2,
            sr_channels=64,
            sr_layers=2,
            vit_patch_size=8,
            vit_embed_dim=256,
            vit_num_heads=4,
            vit_num_layers=2
        )
    
    @pytest.fixture
    def discriminator_config(self):
        """Create test discriminator configuration."""
        return DiscriminatorConfig(
            image_channels=3,
            base_channels=32,
            max_channels=128,
            num_blocks=3,
            use_spectral_norm=True
        )
    
    @pytest.fixture
    def training_config(self):
        """Create test training configuration."""
        return TrainingConfig(
            batch_size=2,
            learning_rate=1e-4,
            num_epochs=1,
            log_interval=1,
            save_interval=1,
            device="cpu",
            mixed_precision=False
        )
    
    @pytest.fixture
    def camera_config(self):
        """Create test camera configuration."""
        return CameraConfig(
            fov=30.0,
            near=0.1,
            far=10.0,
            pitch_range=(-30, 30),
            yaw_range=(-180, 180),
            radius_range=(3.0, 5.0)
        )
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary data directory with test images."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "images"
            data_path.mkdir(parents=True)
            
            # Create dummy images
            for i in range(4):
                img = Image.fromarray(
                    np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
                )
                img.save(data_path / f"image_{i:03d}.jpg")
                
            yield data_path
    
    def test_model_creation_and_forward_pass(self, generator_config, discriminator_config):
        """Test model creation and forward pass."""
        # Create models
        generator = AetheristGenerator(generator_config)
        discriminator = AetheristDiscriminator(discriminator_config)
        
        # Test forward pass
        batch_size = 2
        latent_codes = torch.randn(batch_size, generator_config.latent_dim)
        camera_params = torch.randn(batch_size, 16)  # 4x4 camera matrix flattened
        
        # Generator forward pass
        with torch.no_grad():
            generated_images = generator(latent_codes, camera_params)
            
        assert generated_images.shape == (batch_size, 3, 128, 128)
        assert not torch.isnan(generated_images).any()
        assert not torch.isinf(generated_images).any()
        
        # Discriminator forward pass
        with torch.no_grad():
            disc_output = discriminator(generated_images, camera_params)
            
        assert "validity" in disc_output
        assert "3d_consistency" in disc_output
        assert disc_output["validity"].shape[0] == batch_size
        
    def test_dataset_loading(self, temp_data_dir, camera_config):
        """Test dataset loading and batching."""
        dataset = AetheristDataset(
            data_path=str(temp_data_dir),
            image_size=128,
            camera_config=camera_config,
            augment=True
        )
        
        assert len(dataset) == 4
        
        # Test single item
        item = dataset[0]
        assert "image" in item
        assert "camera_params" in item
        assert item["image"].shape == (3, 128, 128)
        assert item["camera_params"].shape == (16,)
        
        # Test dataloader
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        batch = next(iter(dataloader))
        assert batch["image"].shape == (2, 3, 128, 128)
        assert batch["camera_params"].shape == (2, 16)
        
    def test_training_step(self, generator_config, discriminator_config, 
                          training_config, temp_data_dir, camera_config):
        """Test single training step."""
        # Create models
        generator = AetheristGenerator(generator_config)
        discriminator = AetheristDiscriminator(discriminator_config)
        
        # Create dataset
        dataset = AetheristDataset(
            data_path=str(temp_data_dir),
            image_size=128,
            camera_config=camera_config,
            augment=False
        )
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=training_config.batch_size)
        
        # Create trainer
        trainer = AetheristTrainer(
            generator=generator,
            discriminator=discriminator,
            config=training_config,
            train_loader=dataloader,
            val_loader=None
        )
        
        # Test single step
        batch = next(iter(dataloader))
        losses = trainer._train_step(batch)
        
        assert "generator_loss" in losses
        assert "discriminator_loss" in losses
        assert "adversarial_loss" in losses
        assert "reconstruction_loss" in losses
        
        for loss_name, loss_value in losses.items():
            assert isinstance(loss_value, (float, torch.Tensor))
            if isinstance(loss_value, torch.Tensor):
                assert not torch.isnan(loss_value)
                assert not torch.isinf(loss_value)
                
    def test_inference_pipeline(self, generator_config):
        """Test inference pipeline."""
        generator = AetheristGenerator(generator_config)
        
        # Create inference pipeline
        pipeline = InferencePipeline(
            generator=generator,
            device="cpu"
        )
        
        # Test generation
        images = pipeline.generate(
            num_samples=2,
            seed=42,
            camera_angles=[(0, 0), (30, 45)]
        )
        
        assert len(images) == 2
        for img in images:
            assert isinstance(img, Image.Image)
            assert img.size == (128, 128)
            
    def test_checkpoint_saving_loading(self, generator_config, discriminator_config):
        """Test checkpoint saving and loading."""
        generator = AetheristGenerator(generator_config)
        discriminator = AetheristDiscriminator(discriminator_config)
        
        # Get initial state
        gen_state = generator.state_dict()
        disc_state = discriminator.state_dict()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint.pt"
            
            # Save checkpoint
            checkpoint = {
                "generator": gen_state,
                "discriminator": disc_state,
                "epoch": 1,
                "step": 100,
                "config": {
                    "generator": generator_config.__dict__,
                    "discriminator": discriminator_config.__dict__
                }
            }
            
            torch.save(checkpoint, checkpoint_path)
            
            # Load checkpoint
            loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu")
            
            # Verify contents
            assert "generator" in loaded_checkpoint
            assert "discriminator" in loaded_checkpoint
            assert loaded_checkpoint["epoch"] == 1
            assert loaded_checkpoint["step"] == 100
            
            # Load into new models
            new_generator = AetheristGenerator(generator_config)
            new_discriminator = AetheristDiscriminator(discriminator_config)
            
            new_generator.load_state_dict(loaded_checkpoint["generator"])
            new_discriminator.load_state_dict(loaded_checkpoint["discriminator"])
            
            # Verify models produce same output
            test_input = torch.randn(1, generator_config.latent_dim)
            test_camera = torch.randn(1, 16)
            
            with torch.no_grad():
                orig_output = generator(test_input, test_camera)
                new_output = new_generator(test_input, test_camera)
                
            torch.testing.assert_close(orig_output, new_output)
            
    def test_memory_usage(self, generator_config, discriminator_config):
        """Test memory usage stays reasonable."""
        generator = AetheristGenerator(generator_config)
        discriminator = AetheristDiscriminator(discriminator_config)
        
        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            latent_codes = torch.randn(batch_size, generator_config.latent_dim)
            camera_params = torch.randn(batch_size, 16)
            
            with torch.no_grad():
                # Generator forward
                gen_output = generator(latent_codes, camera_params)
                
                # Discriminator forward
                disc_output = discriminator(gen_output, camera_params)
                
                # Verify outputs
                assert gen_output.shape == (batch_size, 3, 128, 128)
                assert disc_output["validity"].shape[0] == batch_size
                
                # Check memory usage is reasonable (heuristic)
                if hasattr(torch.cuda, 'memory_allocated'):
                    memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                    assert memory_mb < 1000, f"Memory usage too high: {memory_mb:.1f} MB"
                    
    def test_gradient_flow(self, generator_config, discriminator_config):
        """Test gradient flow through models."""
        generator = AetheristGenerator(generator_config)
        discriminator = AetheristDiscriminator(discriminator_config)
        
        # Enable gradients
        generator.train()
        discriminator.train()
        
        batch_size = 2
        latent_codes = torch.randn(batch_size, generator_config.latent_dim, requires_grad=True)
        camera_params = torch.randn(batch_size, 16)
        target_images = torch.randn(batch_size, 3, 128, 128)
        
        # Forward pass
        generated_images = generator(latent_codes, camera_params)
        disc_output = discriminator(generated_images, camera_params)
        
        # Compute dummy loss
        gen_loss = torch.nn.functional.mse_loss(generated_images, target_images)
        disc_loss = torch.nn.functional.mse_loss(
            disc_output["validity"], 
            torch.ones_like(disc_output["validity"])
        )
        
        # Backward pass
        gen_loss.backward(retain_graph=True)
        disc_loss.backward()
        
        # Check gradients exist
        assert latent_codes.grad is not None
        assert not torch.isnan(latent_codes.grad).any()
        
        # Check model gradients
        for name, param in generator.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                
        for name, param in discriminator.named_parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    @pytest.mark.slow
    def test_inference_speed(self):
        """Benchmark inference speed."""
        config = GeneratorConfig(
            latent_dim=256,
            triplane_dim=256,
            triplane_res=64
        )
        
        generator = AetheristGenerator(config)
        generator.eval()
        
        batch_size = 4
        latent_codes = torch.randn(batch_size, config.latent_dim)
        camera_params = torch.randn(batch_size, 16)
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = generator(latent_codes, camera_params)
        
        # Benchmark
        import time
        num_runs = 20
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_runs):
                _ = generator(latent_codes, camera_params)
                
        end_time = time.time()
        
        avg_time_per_batch = (end_time - start_time) / num_runs
        avg_time_per_sample = avg_time_per_batch / batch_size
        
        print(f"\nInference Performance:")
        print(f"  Batch size: {batch_size}")
        print(f"  Average time per batch: {avg_time_per_batch:.3f}s")
        print(f"  Average time per sample: {avg_time_per_sample:.3f}s")
        print(f"  Throughput: {1/avg_time_per_sample:.1f} samples/sec")
        
        # Performance assertion (adjust based on hardware)
        assert avg_time_per_sample < 5.0, "Inference too slow"
        
    @pytest.mark.slow
    def test_memory_scaling(self):
        """Test memory usage scales reasonably with batch size."""
        config = GeneratorConfig(
            latent_dim=256,
            triplane_dim=128,
            triplane_res=64
        )
        
        generator = AetheristGenerator(config)
        
        memory_usage = {}
        
        for batch_size in [1, 2, 4, 8]:
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
                
            latent_codes = torch.randn(batch_size, config.latent_dim)
            camera_params = torch.randn(batch_size, 16)
            
            with torch.no_grad():
                _ = generator(latent_codes, camera_params)
                
            if hasattr(torch.cuda, 'memory_allocated'):
                memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                memory_usage[batch_size] = memory_mb
                
        if memory_usage:
            print(f"\nMemory Scaling:")
            for bs, mem in memory_usage.items():
                print(f"  Batch size {bs}: {mem:.1f} MB")
                
            # Check memory scaling is sub-quadratic
            if len(memory_usage) >= 2:
                ratios = []
                batch_sizes = sorted(memory_usage.keys())
                for i in range(1, len(batch_sizes)):
                    bs1, bs2 = batch_sizes[i-1], batch_sizes[i]
                    mem1, mem2 = memory_usage[bs1], memory_usage[bs2]
                    ratio = (mem2 / mem1) / (bs2 / bs1)
                    ratios.append(ratio)
                    
                avg_ratio = sum(ratios) / len(ratios)
                assert avg_ratio < 2.0, f"Memory scaling too aggressive: {avg_ratio:.2f}"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
