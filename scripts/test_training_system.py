#!/usr/bin/env python3
"""
Test script for the complete Aetherist training system.

This script validates the full training pipeline integration between
generator, discriminator, losses, and optimization.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from src.training.training_loop import AetheristTrainer
from src.models.generator import AetheristGenerator  
from src.models.discriminator import AetheristDiscriminator


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing training pipeline."""
    
    def __init__(self, size: int = 1000, image_size: int = 256):
        """
        Initialize synthetic dataset.
        
        Args:
            size: Number of samples
            image_size: Size of synthetic images
        """
        self.size = size
        self.image_size = image_size
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate synthetic image with some structure
        # This creates colorful patterns that are more interesting than pure noise
        
        # Create gradient patterns
        x = torch.linspace(-1, 1, self.image_size)
        y = torch.linspace(-1, 1, self.image_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Create interesting patterns
        pattern1 = torch.sin(5 * X) * torch.cos(5 * Y)
        pattern2 = torch.sin(3 * (X**2 + Y**2)**0.5)
        pattern3 = torch.cos(4 * X) * torch.sin(6 * Y)
        
        # Combine patterns with random weights
        weights = torch.randn(3) * 0.5
        combined = weights[0] * pattern1 + weights[1] * pattern2 + weights[2] * pattern3
        
        # Add some randomness and normalize
        noise = torch.randn_like(combined) * 0.1
        image = combined + noise
        
        # Normalize to [-1, 1]
        image = torch.tanh(image)
        
        # Convert to RGB
        image = image.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
        
        return image


def test_training_integration():
    """Test the complete training system integration."""
    print("🏗️ Testing Aetherist Training System Integration")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Configuration
    batch_size = 4
    num_epochs = 2
    dataset_size = 20  # Small for testing
    image_size = 128   # Smaller for faster testing
    
    print(f"Batch size: {batch_size}")
    print(f"Training epochs: {num_epochs}")
    print(f"Dataset size: {dataset_size}")
    print(f"Image size: {image_size}x{image_size}")
    print()
    
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    dataset = SyntheticDataset(size=dataset_size, image_size=image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f"Dataset created with {len(dataset)} samples")
    print(f"DataLoader created with {len(dataloader)} batches")
    print()
    
    # Test data loading
    print("Testing data loading...")
    sample_batch = next(iter(dataloader))
    print(f"Sample batch shape: {sample_batch.shape}")
    print(f"Sample batch range: [{sample_batch.min():.3f}, {sample_batch.max():.3f}]")
    
    # Visualize sample
    sample_image = sample_batch[0]  # (3, H, W)
    sample_vis = torch.clamp((sample_image + 1) / 2, 0, 1)  # Convert to [0, 1]
    
    plt.figure(figsize=(6, 6))
    plt.imshow(sample_vis.permute(1, 2, 0).numpy())
    plt.title("Sample Synthetic Image")
    plt.axis('off')
    plt.tight_layout()
    
    # Create output directory
    output_dir = Path("outputs/training_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_dir / "sample_data.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sample data visualization saved to: {output_dir / 'sample_data.png'}")
    print()
    
    # Create trainer with smaller models for testing
    print("Creating models and trainer...")
    
    # Create models directly
    generator = AetheristGenerator(
        latent_dim=256,
        vit_dim=256,
        vit_layers=4,
        triplane_resolution=32,
        triplane_channels=16,
    ).to(device)
    
    discriminator = AetheristDiscriminator(
        input_size=image_size,
        input_channels=3,
        base_channels=32,
        feature_dim=128,
        use_multiscale=False,
        use_consistency_branch=False,  # Disable for single-view testing
        num_views=1,                   # Single view for testing
    ).to(device)
    
    trainer = AetheristTrainer(
        generator=generator,
        discriminator=discriminator,
        device=device,
        adversarial_loss_type="non_saturating",
        lambda_perceptual=0.1,
        lambda_consistency=0.05,
        lambda_r1=10.0,
        generator_lr=2e-4,
        discriminator_lr=2e-4,
        num_views=1,               # Single view for testing
        view_probability=0.0,      # Disable multi-view
        log_every=2,
        save_every=10,
        sample_every=5,
    )
    
    print("Trainer created successfully!")
    print()
    
    # Test single training step
    print("Testing single training step...")
    trainer.generator.train()
    trainer.discriminator.train()
    
    # Wrap the batch in the expected format
    batch_dict = {'images': sample_batch}
    step_metrics = trainer.train_step(batch_dict)
    
    print("Single step metrics:")
    for key, value in step_metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    print()
    
    # Test loss functions directly
    print("Testing loss function components...")
    
    # Generate fake data for loss testing
    with torch.no_grad():
        z = torch.randn(batch_size, 256, device=device)  # latent_dim=256
        
        # Sample camera poses
        from src.utils.camera import sample_camera_poses, perspective_projection_matrix
        eye_positions, view_matrices, camera_angles = sample_camera_poses(
            batch_size, device=device
        )
        proj_matrices = perspective_projection_matrix(
            fov_degrees=torch.tensor([50.0] * batch_size),
            aspect_ratio=torch.tensor([1.0] * batch_size),
        ).to(device)
        camera_matrices = torch.bmm(proj_matrices, view_matrices)
        
        # Generate samples
        fake_output = trainer.generator(z, camera_matrices, return_triplanes=True)
        fake_images = fake_output['high_res_image']
        
        print(f"Generated samples shape: {fake_images.shape}")
        print(f"Generated samples range: [{fake_images.min():.3f}, {fake_images.max():.3f}]")
        
        # Test discriminator
        real_d_output = trainer.discriminator(sample_batch.to(device))
        fake_d_output = trainer.discriminator(fake_images)
        
        print(f"Real discriminator score: {real_d_output['total_score'].mean():.4f}")
        print(f"Fake discriminator score: {fake_d_output['total_score'].mean():.4f}")
    
    print()
    
    # Test short training run
    print(f"Running short training for {num_epochs} epochs...")
    print("=" * 40)
    
    start_time = time.time()
    trainer.train(dataloader, num_epochs=num_epochs)
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f}s")
    print(f"Average time per step: {training_time / (num_epochs * len(dataloader)):.3f}s")
    print()
    
    # Verify checkpoints and samples were created
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    samples = list(sample_dir.glob("*.png"))
    
    print("Training artifacts:")
    print(f"  Checkpoints created: {len(checkpoints)}")
    print(f"  Sample images created: {len(samples)}")
    
    if checkpoints:
        print(f"  Latest checkpoint: {checkpoints[-1].name}")
    if samples:
        print(f"  Latest samples: {samples[-1].name}")
    
    print()
    
    # Test checkpoint loading
    if checkpoints:
        print("Testing checkpoint loading...")
        latest_checkpoint = checkpoints[-1]
        
        # Create new trainer
        trainer2 = AetheristTrainer(
            generator=AetheristGenerator(
                latent_dim=256,
                vit_dim=256,
                vit_layers=4,
                triplane_resolution=32,
                triplane_channels=16,
            ).to(device),
            discriminator=AetheristDiscriminator(
                input_size=image_size,
                input_channels=3,
                base_channels=32,
                feature_dim=128,
                use_multiscale=False,
                use_consistency_branch=False,  # Disable for single-view testing  
                num_views=1,                   # Single view for testing
            ).to(device),
            device=device,
        )
        
        # Load checkpoint
        checkpoint_data = trainer2.checkpoint_manager.load_checkpoint(
            latest_checkpoint,
            trainer2.generator,
            trainer2.discriminator,
            trainer2.g_optimizer,
            trainer2.d_optimizer,
        )
        
        print(f"Successfully loaded checkpoint from step {checkpoint_data['step']}")
        print()
    
    # Generate final samples
    print("Generating final samples...")
    trainer.generate_samples()
    print("Final samples generated")
    print()
    
    # Validation checks
    print("🔍 Validation Checks...")
    checks_passed = 0
    total_checks = 7
    
    # Check 1: Models can forward pass
    try:
        with torch.no_grad():
            z_test = torch.randn(2, 256, device=device)  # latent_dim=256
            camera_test = torch.eye(4, device=device).unsqueeze(0).repeat(2, 1, 1)
            gen_output = trainer.generator(z_test, camera_test)
            disc_output = trainer.discriminator(gen_output['high_res_image'])
        print("✅ Models forward pass working")
        checks_passed += 1
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
    
    # Check 2: Training step completes
    try:
        test_batch = next(iter(dataloader))
        test_batch_dict = {'images': test_batch}
        step_metrics = trainer.train_step(test_batch_dict)
        assert len(step_metrics) > 0
        print("✅ Training step working")
        checks_passed += 1
    except Exception as e:
        print(f"❌ Training step failed: {e}")
    
    # Check 3: Losses are reasonable
    loss_values = [v for k, v in step_metrics.items() if 'loss' in k and isinstance(v, float)]
    if loss_values and all(0.001 < abs(v) < 100 for v in loss_values):
        print("✅ Loss values in reasonable range")
        checks_passed += 1
    else:
        print(f"❌ Loss values out of reasonable range: {loss_values}")
    
    # Check 4: Gradients are flowing
    trainer.g_optimizer.zero_grad()
    trainer.d_optimizer.zero_grad()
    
    step_metrics = trainer.train_step(test_batch)
    
    g_grad_norm = sum(p.grad.norm().item() ** 2 for p in trainer.generator.parameters() if p.grad is not None) ** 0.5
    d_grad_norm = sum(p.grad.norm().item() ** 2 for p in trainer.discriminator.parameters() if p.grad is not None) ** 0.5
    
    if g_grad_norm > 1e-6 and d_grad_norm > 1e-6:
        print("✅ Gradients flowing properly")
        checks_passed += 1
    else:
        print(f"❌ Gradient flow issues: G={g_grad_norm:.6f}, D={d_grad_norm:.6f}")
    
    # Check 5: Checkpoints created and loadable
    if len(checkpoints) > 0:
        try:
            torch.load(checkpoints[-1], map_location='cpu')
            print("✅ Checkpoints created and loadable")
            checks_passed += 1
        except Exception as e:
            print(f"❌ Checkpoint loading failed: {e}")
    else:
        print("❌ No checkpoints created")
    
    # Check 6: Samples generated
    if len(samples) > 0:
        print("✅ Sample generation working")
        checks_passed += 1
    else:
        print("❌ No samples generated")
    
    # Check 7: Learning rates updating
    initial_g_lr = trainer.g_optimizer.param_groups[0]['lr']
    trainer.g_scheduler.step()
    new_g_lr = trainer.g_optimizer.param_groups[0]['lr']
    
    if initial_g_lr != new_g_lr:
        print("✅ Learning rate scheduling working")
        checks_passed += 1
    else:
        print("❌ Learning rate not updating")
    
    print()
    print(f"Validation Summary: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("🎉 ALL TESTS PASSED! Aetherist training system is working correctly!")
        return True
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return False


def benchmark_training_speed():
    """Benchmark training speed."""
    print("\n⚡ Benchmarking Training Speed")
    print("=" * 40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Small models for speed testing
    generator = AetheristGenerator(
        latent_dim=256,
        vit_dim=128,
        vit_layers=2,
        triplane_resolution=16,
        triplane_channels=8,
    ).to(device)
    
    discriminator = AetheristDiscriminator(
        input_size=64,
        input_channels=3,
        base_channels=16,
        feature_dim=64,
        use_multiscale=False,
        use_consistency_branch=False,
        num_views=2,
    ).to(device)
    
    trainer = AetheristTrainer(
        generator=generator,
        discriminator=discriminator,
        device=device,
    )
    
    # Create test batch
    test_batch = torch.randn(4, 3, 64, 64, device=device)
    test_batch_dict = {'images': test_batch}
    
    # Warmup
    trainer.train_step(test_batch_dict)
    
    # Benchmark
    import time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    num_steps = 10
    
    for _ in range(num_steps):
        trainer.train_step(test_batch_dict)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_steps
    
    print(f"Average training step time: {avg_time:.3f}s")
    print(f"Training speed: {1/avg_time:.1f} steps/sec")


if __name__ == "__main__":
    import time
    
    success = test_training_integration()
    
    if success:
        benchmark_training_speed()
    
    print("\n🏁 Training system test complete!")