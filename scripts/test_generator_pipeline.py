#!/usr/bin/env python3
"""
Test script for the complete Aetherist Generator pipeline.

This script validates the full generation process from latent codes to high-resolution images,
testing all components in integration.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from src.models.generator import AetheristGenerator
from src.utils.camera import sample_camera_poses, perspective_projection_matrix


def test_full_pipeline():
    """Test the complete Aetherist Generator pipeline."""
    print("🚀 Testing Aetherist Generator Full Pipeline")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Configuration
    batch_size = 4
    latent_dim = 512
    triplane_resolution = 64
    triplane_channels = 32
    vit_dim = 256
    vit_layers = 6
    low_res = 64
    high_res = 256
    
    print(f"Batch size: {batch_size}")
    print(f"Latent dimension: {latent_dim}")
    print(f"Tri-plane resolution: {triplane_resolution}x{triplane_resolution}")
    print(f"Output resolution: {low_res}x{low_res} → {high_res}x{high_res}")
    print()
    
    # Initialize generator
    print("📦 Initializing Generator...")
    generator = AetheristGenerator(
        vit_dim=vit_dim,
        vit_layers=vit_layers,
        triplane_resolution=triplane_resolution,
        triplane_channels=triplane_channels,
    ).to(device)
    
    total_params = sum(p.numel() for p in generator.parameters())
    trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Generate random inputs
    print("🎲 Generating Random Inputs...")
    z = torch.randn(batch_size, latent_dim, device=device)
    
    # Sample camera poses
    eye_positions, view_matrices, camera_angles = sample_camera_poses(
        batch_size, device=device
    )
    
    # Create projection matrices  
    proj_matrices = perspective_projection_matrix(
        fov_degrees=torch.tensor([50.0] * batch_size),
        aspect_ratio=torch.tensor([1.0] * batch_size),
    ).to(device)
    
    # Combine view and projection matrices
    camera_matrices = torch.bmm(proj_matrices, view_matrices)
    
    print(f"Latent codes: {z.shape}")
    print(f"Camera matrices: {camera_matrices.shape}")
    print(f"Camera angles: {camera_angles.shape}")
    print()
    
    # Forward pass
    print("⚡ Running Forward Pass...")
    with torch.no_grad():
        output = generator(z, camera_matrices, return_triplanes=True)
    
    print("✅ Forward pass completed!")
    print()
    
    # Analyze outputs
    print("📊 Analyzing Outputs...")
    print(f"Style vectors (w): {output['w'].shape}")
    print(f"ViT features: {output['vit_features'].shape}")
    print(f"Low-res images: {output['low_res_image'].shape}")
    print(f"High-res images: {output['high_res_image'].shape}")
    print()
    
    # Analyze tri-planes
    print("🔍 Tri-plane Analysis...")
    triplanes = output['triplanes']
    for name, plane in triplanes.items():
        print(f"{name}: {plane.shape} | min: {plane.min():.3f} | max: {plane.max():.3f} | mean: {plane.mean():.3f}")
    print()
    
    # Image statistics
    print("🖼️ Image Statistics...")
    low_res_img = output['low_res_image']
    high_res_img = output['high_res_image']
    
    print(f"Low-res image range: [{low_res_img.min():.3f}, {low_res_img.max():.3f}]")
    print(f"High-res image range: [{high_res_img.min():.3f}, {high_res_img.max():.3f}]")
    
    # Convert to [0,1] range for visualization
    low_res_vis = torch.clamp((low_res_img + 1) / 2, 0, 1)
    high_res_vis = torch.clamp((high_res_img + 1) / 2, 0, 1)
    
    print(f"Normalized low-res range: [{low_res_vis.min():.3f}, {low_res_vis.max():.3f}]")
    print(f"Normalized high-res range: [{high_res_vis.min():.3f}, {high_res_vis.max():.3f}]")
    print()
    
    # Memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_cached = torch.cuda.memory_reserved(device) / 1024**3
        print(f"GPU Memory - Allocated: {memory_allocated:.2f} GB | Cached: {memory_cached:.2f} GB")
        print()
    
    # Save sample outputs
    output_dir = Path("outputs/generator_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("💾 Saving Sample Outputs...")
    
    # Save first batch sample
    sample_idx = 0
    
    # Low-res image
    low_res_sample = low_res_vis[sample_idx].cpu().permute(1, 2, 0).numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(low_res_sample)
    plt.title(f"Low-res Output ({low_res}x{low_res})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / "low_res_sample.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # High-res image
    high_res_sample = high_res_vis[sample_idx].cpu().permute(1, 2, 0).numpy()
    plt.figure(figsize=(10, 10))
    plt.imshow(high_res_sample)
    plt.title(f"High-res Output ({high_res}x{high_res})")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_dir / "high_res_sample.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Tri-plane visualizations
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plane_names = ['xy', 'xz', 'yz']
    
    for i, (name, plane) in enumerate(triplanes.items()):
        # Average across channels for visualization
        plane_vis = plane[sample_idx].mean(dim=0).cpu().numpy()
        
        axes[i].imshow(plane_vis, cmap='viridis')
        axes[i].set_title(f"Tri-plane {name.upper()}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "triplanes_sample.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Sample outputs saved to: {output_dir}")
    print()
    
    # Validation checks
    print("🔍 Validation Checks...")
    checks_passed = 0
    total_checks = 6
    
    # Check 1: Output shapes
    if (output['low_res_image'].shape == (batch_size, 3, low_res, low_res) and 
        output['high_res_image'].shape == (batch_size, 3, high_res, high_res)):
        print("✅ Output shapes are correct")
        checks_passed += 1
    else:
        print("❌ Output shapes are incorrect")
    
    # Check 2: Tri-plane shapes
    triplane_shapes_correct = all(
        plane.shape == (batch_size, triplane_channels, triplane_resolution, triplane_resolution)
        for plane in triplanes.values()
    )
    if triplane_shapes_correct:
        print("✅ Tri-plane shapes are correct")
        checks_passed += 1
    else:
        print("❌ Tri-plane shapes are incorrect")
    
    # Check 3: No NaN values
    has_nan = any(torch.isnan(tensor).any() for tensor in [
        output['low_res_image'], output['high_res_image'], output['w']
    ] + list(triplanes.values()))
    if not has_nan:
        print("✅ No NaN values detected")
        checks_passed += 1
    else:
        print("❌ NaN values detected")
    
    # Check 4: Reasonable value ranges
    low_res_range_ok = -2 <= low_res_img.min() <= 2 and -2 <= low_res_img.max() <= 2
    high_res_range_ok = -2 <= high_res_img.min() <= 2 and -2 <= high_res_img.max() <= 2
    if low_res_range_ok and high_res_range_ok:
        print("✅ Image values in reasonable range")
        checks_passed += 1
    else:
        print("❌ Image values out of reasonable range")
    
    # Check 5: Super-resolution improvement
    # Downsample high-res and compare with low-res
    downsampled_high = F.interpolate(high_res_img, size=(low_res, low_res), mode='bilinear')
    mse_improvement = F.mse_loss(low_res_img, downsampled_high)
    if mse_improvement < 1.0:  # Reasonable threshold
        print("✅ Super-resolution provides consistent upsampling")
        checks_passed += 1
    else:
        print("❌ Super-resolution inconsistent with low-res")
    
    # Check 6: Style vector dimensionality
    if output['w'].shape == (batch_size, latent_dim):
        print("✅ Style vector dimensions correct")
        checks_passed += 1
    else:
        print("❌ Style vector dimensions incorrect")
    
    print()
    print(f"Validation Summary: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("🎉 ALL TESTS PASSED! Aetherist Generator pipeline is working correctly!")
        return True
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return False


def benchmark_generation_speed():
    """Benchmark generation speed for different batch sizes."""
    print("\n🏃 Benchmarking Generation Speed")
    print("=" * 40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize generator
    generator = AetheristGenerator(
        vit_dim=256,
        vit_layers=4,
        triplane_resolution=64,
        triplane_channels=32,
    ).to(device)
    
    # Warm up
    with torch.no_grad():
        z_warmup = torch.randn(1, 512, device=device)
        camera_warmup = torch.eye(4, device=device).unsqueeze(0)
        _ = generator(z_warmup, camera_warmup)
    
    # Benchmark different batch sizes
    batch_sizes = [1, 2, 4, 8] if torch.cuda.is_available() else [1, 2]
    
    for batch_size in batch_sizes:
        try:
            z = torch.randn(batch_size, 512, device=device)
            camera_matrices = torch.eye(4, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
            
            # Time generation
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            import time
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(5):  # Average over 5 runs
                    output = generator(z, camera_matrices)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.time()
            avg_time = (end_time - start_time) / 5
            images_per_sec = batch_size / avg_time
            
            print(f"Batch size {batch_size:2d}: {avg_time:.3f}s per batch, {images_per_sec:.1f} images/sec")
            
        except RuntimeError as e:
            print(f"Batch size {batch_size:2d}: Failed - {str(e)}")
            break


if __name__ == "__main__":
    success = test_full_pipeline()
    
    if success:
        benchmark_generation_speed()
    
    print("\n🏁 Generator pipeline test complete!")