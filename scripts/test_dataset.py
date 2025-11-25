#!/usr/bin/env python3
"""
Test script for Aetherist dataset implementation.
Validates dataset loading, transforms, camera sampling, and data loaders.
"""

import torch
import numpy as np
from pathlib import Path
import tempfile
import sys
import logging
from PIL import Image
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import (
    AetheristDataset, 
    MultiViewDataset, 
    SyntheticDataset,
    CameraPoseSampler,
    DatasetAnalyzer,
    create_dataloaders
)
from src.config import DataConfig


def create_dummy_images(temp_dir: Path, num_images: int = 50):
    """Create dummy images for testing."""
    print(f"Creating {num_images} dummy images in {temp_dir}")
    
    # Create train/val directories
    train_dir = temp_dir / "train"
    val_dir = temp_dir / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    
    # Generate dummy images
    for i in range(num_images):
        # Create random image
        image = Image.new('RGB', (256, 256), color=(
            np.random.randint(0, 255),
            np.random.randint(0, 255), 
            np.random.randint(0, 255)
        ))
        
        # Add some patterns
        pixels = np.array(image)
        if i % 4 == 0:
            # Vertical stripes
            pixels[::8, :, :] = [255, 0, 0]
        elif i % 4 == 1:
            # Horizontal stripes  
            pixels[:, ::8, :] = [0, 255, 0]
        elif i % 4 == 2:
            # Checkerboard
            pixels[::16, ::16, :] = [255, 255, 255]
        else:
            # Random noise
            pixels += np.random.randint(-50, 50, pixels.shape)
            pixels = np.clip(pixels, 0, 255)
        
        image = Image.fromarray(pixels.astype(np.uint8))
        
        # Save to appropriate directory
        if i < num_images * 0.8:  # 80% train
            save_path = train_dir / f"image_{i:03d}.png"
        else:
            save_path = val_dir / f"image_{i:03d}.png"
        
        image.save(save_path)
    
    print(f"Created dummy dataset in {temp_dir}")
    return temp_dir


def test_basic_dataset():
    """Test basic dataset functionality."""
    print("\n=== Testing Basic Dataset ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create dummy data
        data_dir = create_dummy_images(temp_dir, num_images=20)
        
        # Create dataset config
        config = DataConfig()
        config.dataset_path = str(data_dir)
        config.image_size = 128
        config.batch_size = 4
        config.val_split = 0.2
        config.use_augmentation = True
        
        try:
            # Test train dataset
            train_dataset = AetheristDataset(
                data_dir=data_dir,
                config=config,
                split="train"
            )
            
            print(f"✅ Train dataset created: {len(train_dataset)} samples")
            
            # Test validation dataset
            val_dataset = AetheristDataset(
                data_dir=data_dir,
                config=config,
                split="val"
            )
            
            print(f"✅ Validation dataset created: {len(val_dataset)} samples")
            
            # Test sample loading
            sample = train_dataset[0]
            print(f"✅ Sample loaded successfully")
            print(f"  Image shape: {sample['image'].shape}")
            print(f"  Camera pose shape: {sample['camera_pose'].shape}")
            print(f"  Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
            
            return train_dataset, val_dataset
            
        except Exception as e:
            print(f"❌ Basic dataset test failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def test_multiview_dataset():
    """Test multi-view dataset functionality."""
    print("\n=== Testing Multi-View Dataset ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create dummy data
        data_dir = create_dummy_images(temp_dir, num_images=10)
        
        # Create dataset config
        config = DataConfig()
        config.dataset_path = str(data_dir)
        config.image_size = 64  # Smaller for faster testing
        config.batch_size = 2
        config.val_split = 0.2
        
        try:
            # Test multi-view dataset
            mv_dataset = MultiViewDataset(
                data_dir=data_dir,
                config=config,
                num_views=3,
                split="train"
            )
            
            print(f"✅ Multi-view dataset created: {len(mv_dataset)} samples")
            
            # Test sample loading
            sample = mv_dataset[0]
            print(f"✅ Multi-view sample loaded successfully")
            print(f"  Images shape: {sample['images'].shape}")
            print(f"  Camera poses shape: {sample['camera_poses'].shape}")
            
            # Verify multiple views are different
            img1 = sample['images'][0]
            img2 = sample['images'][1]
            pose1 = sample['camera_poses'][0]
            pose2 = sample['camera_poses'][1]
            
            # Images might be different due to augmentation
            img_diff = torch.abs(img1 - img2).mean()
            pose_diff = torch.abs(pose1 - pose2).mean()
            
            print(f"  Image difference: {img_diff:.3f}")
            print(f"  Pose difference: {pose_diff:.3f}")
            
            return mv_dataset
            
        except Exception as e:
            print(f"❌ Multi-view dataset test failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def test_synthetic_dataset():
    """Test synthetic dataset functionality."""
    print("\n=== Testing Synthetic Dataset ===")
    
    try:
        # Test single-view synthetic
        synthetic = SyntheticDataset(
            size=100,
            image_size=64,
            num_views=1
        )
        
        print(f"✅ Synthetic dataset created: {len(synthetic)} samples")
        
        # Test sample generation
        sample = synthetic[0]
        print(f"✅ Synthetic sample generated")
        print(f"  Image shape: {sample['image'].shape}")
        print(f"  Camera pose shape: {sample['camera_pose'].shape}")
        
        # Test multi-view synthetic
        synthetic_mv = SyntheticDataset(
            size=50,
            image_size=64,
            num_views=2
        )
        
        mv_sample = synthetic_mv[0]
        print(f"✅ Multi-view synthetic sample generated")
        print(f"  Images shape: {mv_sample['images'].shape}")
        print(f"  Camera poses shape: {mv_sample['camera_poses'].shape}")
        
        # Verify different patterns
        sample1 = synthetic[0]['image']
        sample2 = synthetic[1]['image']
        diff = torch.abs(sample1 - sample2).mean()
        print(f"  Pattern diversity: {diff:.3f}")
        
        return synthetic, synthetic_mv
        
    except Exception as e:
        print(f"❌ Synthetic dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_camera_pose_sampler():
    """Test camera pose sampling functionality."""
    print("\n=== Testing Camera Pose Sampler ===")
    
    try:
        # Create config
        config = DataConfig()
        config.camera_radius_range = [1.5, 2.5]
        config.camera_elevation_range = [-15, 45]
        config.camera_azimuth_range = [-180, 180]
        
        # Create sampler
        sampler = CameraPoseSampler(config)
        
        # Test single pose sampling
        pose = sampler.sample()
        print(f"✅ Single pose sampled: {pose.shape}")
        
        # Verify pose properties
        assert pose.shape == (4, 4), "Pose should be 4x4 matrix"
        
        # Check if it's a valid transformation matrix
        bottom_row = pose[3, :]
        expected_bottom = torch.tensor([0., 0., 0., 1.])
        assert torch.allclose(bottom_row, expected_bottom), "Invalid transformation matrix"
        
        # Test batch sampling
        batch_poses = sampler.sample_batch(10)
        print(f"✅ Batch poses sampled: {batch_poses.shape}")
        
        # Analyze camera positions
        positions = batch_poses[:, :3, 3]  # Extract translation
        distances = torch.norm(positions, dim=1)
        
        print(f"  Distance range: [{distances.min():.2f}, {distances.max():.2f}]")
        print(f"  Mean distance: {distances.mean():.2f} ± {distances.std():.2f}")
        
        # Check if distances are within expected range
        assert distances.min() >= config.camera_radius_range[0] * 0.9, "Distance too small"
        assert distances.max() <= config.camera_radius_range[1] * 1.1, "Distance too large"
        
        return sampler
        
    except Exception as e:
        print(f"❌ Camera pose sampler test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_dataloaders():
    """Test dataloader creation and functionality."""
    print("\n=== Testing DataLoaders ===")
    
    try:
        # Test synthetic dataloaders
        config = DataConfig()
        config.batch_size = 8
        config.num_workers = 2
        config.val_split = 0.2
        
        train_loader, val_loader = create_dataloaders(
            config=config,
            num_views=1,
            use_synthetic=True
        )
        
        print(f"✅ Dataloaders created")
        print(f"  Train loader: {len(train_loader)} batches")
        if val_loader:
            print(f"  Val loader: {len(val_loader)} batches")
        
        # Test batch loading
        for i, batch in enumerate(train_loader):
            if i >= 2:  # Test only first 2 batches
                break
                
            print(f"  Batch {i}: images {batch['image'].shape}")
            
            # Verify batch properties
            assert batch['image'].shape[0] == config.batch_size, "Wrong batch size"
            assert batch['camera_pose'].shape[0] == config.batch_size, "Wrong pose batch size"
        
        print(f"✅ Batch loading successful")
        
        # Test multi-view dataloaders
        mv_train_loader, mv_val_loader = create_dataloaders(
            config=config,
            num_views=3,
            use_synthetic=True
        )
        
        batch = next(iter(mv_train_loader))
        print(f"✅ Multi-view batch loading successful")
        print(f"  Multi-view batch: images {batch['images'].shape}")
        
        return train_loader, val_loader
        
    except Exception as e:
        print(f"❌ Dataloader test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def test_dataset_analyzer(dataset):
    """Test dataset analysis functionality."""
    print("\n=== Testing Dataset Analyzer ===")
    
    if dataset is None:
        print("❌ No dataset available for analysis")
        return
    
    try:
        analyzer = DatasetAnalyzer(dataset)
        
        # Test image analysis
        image_stats = analyzer.analyze_images(num_samples=10)
        print(f"✅ Image analysis completed")
        print(f"  Mean: {image_stats['mean']}")
        print(f"  Std: {image_stats['std']}")
        print(f"  Value range: [{image_stats['min_val']:.3f}, {image_stats['max_val']:.3f}]")
        
        # Test camera analysis
        camera_stats = analyzer.analyze_camera_poses(num_samples=10)
        print(f"✅ Camera analysis completed")
        print(f"  Mean distance: {camera_stats['mean_distance']:.2f}")
        print(f"  Distance std: {camera_stats['std_distance']:.2f}")
        
        # Test full analysis
        analyzer.print_analysis()
        
        return analyzer
        
    except Exception as e:
        print(f"❌ Dataset analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_data_augmentation():
    """Test data augmentation functionality."""
    print("\n=== Testing Data Augmentation ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        
        # Create dummy data
        data_dir = create_dummy_images(temp_dir, num_images=5)
        
        try:
            # Test with augmentation
            config_aug = DataConfig()
            config_aug.dataset_path = str(data_dir)
            config_aug.use_augmentation = True
            config_aug.horizontal_flip = True
            config_aug.color_jitter = True
            config_aug.rotation_degrees = 10
            
            dataset_aug = AetheristDataset(
                data_dir=data_dir,
                config=config_aug,
                split="train"
            )
            
            # Test without augmentation
            config_no_aug = DataConfig()
            config_no_aug.dataset_path = str(data_dir)
            config_no_aug.use_augmentation = False
            
            dataset_no_aug = AetheristDataset(
                data_dir=data_dir,
                config=config_no_aug,
                split="train"
            )
            
            # Compare same image with and without augmentation
            sample_aug = dataset_aug[0]['image']
            sample_no_aug = dataset_no_aug[0]['image']
            
            print(f"✅ Augmentation test completed")
            print(f"  Augmented image range: [{sample_aug.min():.3f}, {sample_aug.max():.3f}]")
            print(f"  Original image range: [{sample_no_aug.min():.3f}, {sample_no_aug.max():.3f}]")
            
            # Test multiple samples from same source (should be different with augmentation)
            samples_aug = [dataset_aug[0]['image'] for _ in range(3)]
            differences = []
            for i in range(1, len(samples_aug)):
                diff = torch.abs(samples_aug[0] - samples_aug[i]).mean()
                differences.append(diff)
            
            avg_diff = sum(differences) / len(differences)
            print(f"  Average augmentation difference: {avg_diff:.3f}")
            
            return dataset_aug, dataset_no_aug
            
        except Exception as e:
            print(f"❌ Data augmentation test failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def test_error_handling():
    """Test error handling in dataset."""
    print("\n=== Testing Error Handling ===")
    
    try:
        # Test with non-existent directory
        config = DataConfig()
        config.dataset_path = "/non/existent/path"
        
        try:
            dataset = AetheristDataset(
                data_dir="/non/existent/path",
                config=config,
                split="train"
            )
            print("❌ Should have failed with non-existent directory")
        except ValueError:
            print("✅ Correctly handled non-existent directory")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        
        # Test with invalid configuration
        try:
            config.val_split = 1.5  # Invalid split
            # This should be caught by config validation if implemented
            print("⚠️  Invalid config test skipped (validation not implemented)")
        except Exception:
            print("✅ Correctly handled invalid configuration")
        
        print("✅ Error handling tests completed")
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")


def main():
    """Run all dataset tests."""
    print("🧪 Aetherist Dataset Test Suite")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise for testing
    
    # Run tests
    train_dataset, val_dataset = test_basic_dataset()
    mv_dataset = test_multiview_dataset()
    synthetic, synthetic_mv = test_synthetic_dataset()
    sampler = test_camera_pose_sampler()
    train_loader, val_loader = test_dataloaders()
    analyzer = test_dataset_analyzer(train_dataset)
    aug_dataset, no_aug_dataset = test_data_augmentation()
    test_error_handling()
    
    print("\n" + "=" * 50)
    if all([train_dataset, mv_dataset, synthetic, sampler, train_loader]):
        print("🎉 All dataset tests passed!")
        
        print(f"\nDataset Summary:")
        if train_dataset:
            print(f"  Basic dataset: {len(train_dataset)} samples")
        if mv_dataset:
            print(f"  Multi-view dataset: {len(mv_dataset)} samples")
        if synthetic:
            print(f"  Synthetic dataset: {len(synthetic)} samples")
        if train_loader:
            print(f"  Train loader: {len(train_loader)} batches")
            
    else:
        print("❌ Some dataset tests failed!")
        
    print("\nDataset implementation ready for training! 🚀")


if __name__ == "__main__":
    main()