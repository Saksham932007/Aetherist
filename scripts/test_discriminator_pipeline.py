#!/usr/bin/env python3
"""
Test script for the Aetherist Discriminator architecture.

This script validates the dual-branch discriminator components and full architecture.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
from pathlib import Path

from src.models.discriminator import (
    AetheristDiscriminator,
    ImageQualityBranch,
    ConsistencyBranch,
    DiscriminatorBlock,
    SpectralNorm,
)


def test_discriminator_pipeline():
    """Test the complete Aetherist Discriminator pipeline."""
    print("🎯 Testing Aetherist Discriminator Architecture")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Configuration
    batch_size = 2
    input_size = 256
    input_channels = 3
    num_views = 2
    
    print(f"Batch size: {batch_size}")
    print(f"Input size: {input_size}x{input_size}")
    print(f"Input channels: {input_channels}")
    print(f"Number of views: {num_views}")
    print()
    
    # Test individual components first
    print("🧩 Testing Individual Components...")
    
    # 1. Test Discriminator Block
    print("Testing DiscriminatorBlock...")
    disc_block = DiscriminatorBlock(
        in_channels=64,
        out_channels=128,
        use_attention=True,
    ).to(device)
    
    test_feature = torch.randn(batch_size, 64, 32, 32, device=device)
    block_output = disc_block(test_feature)
    print(f"Block input: {test_feature.shape} → output: {block_output.shape}")
    assert block_output.shape == (batch_size, 128, 32, 32)
    print("✅ DiscriminatorBlock working correctly")
    print()
    
    # 2. Test Image Quality Branch
    print("Testing Image Quality Branch...")
    quality_branch = ImageQualityBranch(
        input_size=input_size,
        input_channels=input_channels,
        base_channels=64,
        num_layers=5,
    ).to(device)
    
    test_image = torch.randn(batch_size, input_channels, input_size, input_size, device=device)
    quality_output = quality_branch(test_image)
    
    print(f"Quality input: {test_image.shape}")
    print(f"Quality score: {quality_output['quality_score'].shape}")
    print(f"Final feature: {quality_output['final_feature'].shape}")
    print(f"Number of feature layers: {len(quality_output['features'])}")
    
    assert quality_output['quality_score'].shape == (batch_size, 1)
    print("✅ Image Quality Branch working correctly")
    print()
    
    # 3. Test 3D Consistency Branch
    print("Testing 3D Consistency Branch...")
    consistency_branch = ConsistencyBranch(
        input_size=input_size,
        input_channels=input_channels,
        feature_dim=256,
        num_views=num_views,
        use_cross_attention=True,
    ).to(device)
    
    test_views = [
        torch.randn(batch_size, input_channels, input_size, input_size, device=device)
        for _ in range(num_views)
    ]
    
    consistency_output = consistency_branch(test_views)
    
    print(f"Consistency views: {[view.shape for view in test_views]}")
    print(f"Consistency score: {consistency_output['consistency_score'].shape}")
    print(f"Aggregated feature: {consistency_output['aggregated_feature'].shape}")
    print(f"View features: {len(consistency_output['view_features'])} views")
    
    assert consistency_output['consistency_score'].shape == (batch_size, 1)
    print("✅ 3D Consistency Branch working correctly")
    print()
    
    # 4. Test Full Discriminator Architecture
    print("🏗️ Testing Full Discriminator Architecture...")
    discriminator = AetheristDiscriminator(
        input_size=input_size,
        input_channels=input_channels,
        base_channels=64,
        feature_dim=256,
        use_multiscale=True,
        scales=[256, 128, 64],
        num_views=num_views,
        use_consistency_branch=True,
        lambda_consistency=0.1,
    ).to(device)
    
    # Parameter count
    total_params = sum(p.numel() for p in discriminator.parameters())
    trainable_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    # Test with single image
    print("Testing Single Image Analysis...")
    single_output = discriminator(test_image, return_features=True)
    
    print("Single Image Results:")
    print(f"  Quality score: {single_output['quality_score'].shape}")
    print(f"  Final score: {single_output['final_score'].shape}")
    print(f"  Total score: {single_output['total_score'].shape}")
    
    # Value ranges
    print(f"  Quality score range: [{single_output['quality_score'].min():.3f}, {single_output['quality_score'].max():.3f}]")
    print(f"  Final score range: [{single_output['final_score'].min():.3f}, {single_output['final_score'].max():.3f}]")
    print()
    
    # Test with multiple views for consistency analysis
    print("Testing Multi-view Consistency Analysis...")
    multi_output = discriminator(test_views, return_features=True)
    
    print("Multi-view Results:")
    print(f"  Quality score: {multi_output['quality_score'].shape}")
    print(f"  Consistency score: {multi_output['consistency_score'].shape}")
    print(f"  Final score: {multi_output['final_score'].shape}")
    print(f"  Total score: {multi_output['total_score'].shape}")
    
    # Value ranges
    print(f"  Quality score range: [{multi_output['quality_score'].min():.3f}, {multi_output['quality_score'].max():.3f}]")
    print(f"  Consistency score range: [{multi_output['consistency_score'].min():.3f}, {multi_output['consistency_score'].max():.3f}]")
    print(f"  Final score range: [{multi_output['final_score'].min():.3f}, {multi_output['final_score'].max():.3f}]")
    print()
    
    # Test gradient flow
    print("Testing Gradient Flow...")
    loss = multi_output['total_score'].mean()
    loss.backward()
    
    # Check if gradients are flowing
    grad_count = 0
    total_grad_norm = 0.0
    
    for name, param in discriminator.named_parameters():
        if param.grad is not None:
            grad_count += 1
            total_grad_norm += param.grad.norm().item() ** 2
        
    total_grad_norm = total_grad_norm ** 0.5
    
    print(f"Parameters with gradients: {grad_count}")
    print(f"Total gradient norm: {total_grad_norm:.6f}")
    print()
    
    # Memory usage
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
        memory_cached = torch.cuda.memory_reserved(device) / 1024**3
        print(f"GPU Memory - Allocated: {memory_allocated:.2f} GB | Cached: {memory_cached:.2f} GB")
        print()
    
    # Validation checks
    print("🔍 Validation Checks...")
    checks_passed = 0
    total_checks = 8
    
    # Check 1: Output shapes
    if (single_output['quality_score'].shape == (batch_size, 1) and 
        single_output['final_score'].shape == (batch_size, 1)):
        print("✅ Single image output shapes correct")
        checks_passed += 1
    else:
        print("❌ Single image output shapes incorrect")
    
    # Check 2: Multi-view output shapes
    if (multi_output['quality_score'].shape == (batch_size, 1) and 
        multi_output['consistency_score'].shape == (batch_size, 1) and
        multi_output['final_score'].shape == (batch_size, 1)):
        print("✅ Multi-view output shapes correct")
        checks_passed += 1
    else:
        print("❌ Multi-view output shapes incorrect")
    
    # Check 3: No NaN values
    has_nan = any(torch.isnan(tensor).any() for tensor in [
        single_output['quality_score'], single_output['final_score'],
        multi_output['quality_score'], multi_output['consistency_score'], multi_output['final_score']
    ])
    if not has_nan:
        print("✅ No NaN values detected")
        checks_passed += 1
    else:
        print("❌ NaN values detected")
    
    # Check 4: Reasonable score ranges
    quality_range_ok = -10 <= single_output['quality_score'].min() <= 10 and -10 <= single_output['quality_score'].max() <= 10
    if quality_range_ok:
        print("✅ Quality scores in reasonable range")
        checks_passed += 1
    else:
        print("❌ Quality scores out of reasonable range")
    
    # Check 5: Gradient flow
    if grad_count > 0 and total_grad_norm > 1e-8:
        print("✅ Gradients flowing properly")
        checks_passed += 1
    else:
        print("❌ Gradient flow issues")
    
    # Check 6: Spectral normalization working
    spectral_norm_count = 0
    for module in discriminator.modules():
        if hasattr(module, 'weight_u') and hasattr(module, 'weight_v'):
            spectral_norm_count += 1
    
    if spectral_norm_count > 0:
        print("✅ Spectral normalization layers present")
        checks_passed += 1
    else:
        print("❌ No spectral normalization found")
    
    # Check 7: Feature extraction
    if ('quality_features' in single_output and 
        len(single_output['quality_features']) > 0):
        print("✅ Feature extraction working")
        checks_passed += 1
    else:
        print("❌ Feature extraction not working")
    
    # Check 8: Consistency analysis
    if ('consistency_score' in multi_output and 
        'consistency_features' in multi_output):
        print("✅ Consistency analysis working")
        checks_passed += 1
    else:
        print("❌ Consistency analysis not working")
    
    print()
    print(f"Validation Summary: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("🎉 ALL TESTS PASSED! Aetherist Discriminator architecture is working correctly!")
        return True
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return False


def benchmark_discriminator_speed():
    """Benchmark discriminator inference speed."""
    print("\n⚡ Benchmarking Discriminator Speed")
    print("=" * 40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize discriminator
    discriminator = AetheristDiscriminator(
        input_size=256,
        input_channels=3,
        base_channels=64,
        use_multiscale=True,
        use_consistency_branch=True,
        num_views=2,
    ).to(device)
    
    # Warm up
    with torch.no_grad():
        warmup_image = torch.randn(1, 3, 256, 256, device=device)
        _ = discriminator(warmup_image)
    
    # Benchmark different batch sizes
    batch_sizes = [1, 2, 4] if torch.cuda.is_available() else [1, 2]
    
    for batch_size in batch_sizes:
        try:
            test_image = torch.randn(batch_size, 3, 256, 256, device=device)
            
            # Time discrimination
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            import time
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(5):  # Average over 5 runs
                    output = discriminator(test_image)
            
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
    success = test_discriminator_pipeline()
    
    if success:
        benchmark_discriminator_speed()
    
    print("\n🏁 Discriminator architecture test complete!")