"""
Dataset implementation for Aetherist training.
Handles image loading, camera pose sampling, and data augmentation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
import random

from ..utils.camera_utils import sample_camera_poses, create_camera_pose
from ..config.config_manager import DataConfig


logger = logging.getLogger(__name__)


class AetheristDataset(Dataset):
    """
    Dataset for Aetherist training.
    
    Supports:
    - Image loading with various formats
    - Camera pose sampling for 3D-aware training
    - Data augmentation and preprocessing
    - Multi-view consistency training
    - Validation split handling
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        config: DataConfig,
        split: str = "train",
        transforms_list: Optional[List] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Directory containing images
            config: Data configuration
            split: Dataset split ("train", "val", "test")
            transforms_list: Custom transforms (if None, uses config)
        """
        self.data_dir = Path(data_dir)
        self.config = config
        self.split = split
        
        # Find image files
        self.image_paths = self._find_images()
        
        # Create train/validation split
        self.image_paths = self._create_split()
        
        # Setup transforms
        self.transforms = self._create_transforms(transforms_list)
        
        # Camera pose sampling
        self.camera_sampler = CameraPoseSampler(config)
        
        logger.info(f"Created {split} dataset with {len(self.image_paths)} images")
    
    def _find_images(self) -> List[Path]:
        """Find all image files in the dataset directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = []
        
        # Support nested directory structure
        if (self.data_dir / self.split).exists():
            search_dir = self.data_dir / self.split
        else:
            search_dir = self.data_dir
        
        for ext in image_extensions:
            image_paths.extend(search_dir.rglob(f"*{ext}"))
            image_paths.extend(search_dir.rglob(f"*{ext.upper()}"))
        
        if not image_paths:
            raise ValueError(f"No images found in {search_dir}")
        
        return sorted(image_paths)
    
    def _create_split(self) -> List[Path]:
        """Create train/validation split."""
        if self.split == "test":
            # For test split, use all available images
            return self.image_paths
        
        # Create reproducible split
        random.seed(self.config.val_seed)
        shuffled_paths = self.image_paths.copy()
        random.shuffle(shuffled_paths)
        
        # Calculate split point
        val_size = int(len(shuffled_paths) * self.config.val_split)
        
        if self.split == "train":
            return shuffled_paths[val_size:]
        elif self.split == "val":
            return shuffled_paths[:val_size]
        else:
            raise ValueError(f"Unknown split: {self.split}")
    
    def _create_transforms(self, transforms_list: Optional[List]) -> transforms.Compose:
        """Create data transforms based on configuration."""
        if transforms_list is not None:
            return transforms.Compose(transforms_list)
        
        transform_list = []
        
        # Resize to target size
        transform_list.append(transforms.Resize(
            (self.config.image_size, self.config.image_size),
            interpolation=transforms.InterpolationMode.BILINEAR
        ))
        
        # Data augmentation for training
        if self.split == "train" and self.config.use_augmentation:
            
            # Horizontal flip
            if self.config.horizontal_flip:
                transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
            
            # Color jitter
            if self.config.color_jitter:
                transform_list.append(transforms.ColorJitter(
                    brightness=0.1,
                    contrast=0.1,
                    saturation=0.1,
                    hue=0.05
                ))
            
            # Random rotation
            if self.config.rotation_degrees > 0:
                transform_list.append(transforms.RandomRotation(
                    degrees=self.config.rotation_degrees,
                    interpolation=transforms.InterpolationMode.BILINEAR
                ))
        
        # Convert to tensor and normalize
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
        ])
        
        return transforms.Compose(transform_list)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item.
        
        Returns:
            Dictionary containing:
            - "image": Transformed image tensor [3, H, W]
            - "camera_pose": Camera pose matrix [4, 4]
            - "path": Image path (for debugging)
            - "index": Sample index
        """
        # Load image
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            # Return a black image as fallback
            image = Image.new("RGB", (self.config.image_size, self.config.image_size))
        
        # Apply transforms
        image_tensor = self.transforms(image)
        
        # Sample camera pose
        camera_pose = self.camera_sampler.sample()
        
        return {
            "image": image_tensor,
            "camera_pose": camera_pose,
            "path": str(image_path),
            "index": idx,
        }


class CameraPoseSampler:
    """
    Camera pose sampler for 3D-aware training.
    Samples camera positions from spherical coordinates.
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    def sample(self, device: str = "cpu") -> torch.Tensor:
        """
        Sample a random camera pose.
        
        Args:
            device: Target device for the tensor
            
        Returns:
            Camera pose matrix [4, 4]
        """
        # Sample spherical coordinates
        radius = np.random.uniform(*self.config.camera_radius_range)
        elevation = np.random.uniform(*self.config.camera_elevation_range)
        azimuth = np.random.uniform(*self.config.camera_azimuth_range)
        
        # Convert to radians
        elevation_rad = np.deg2rad(elevation)
        azimuth_rad = np.deg2rad(azimuth)
        
        # Create camera pose
        camera_pose = create_camera_pose(
            elevation=elevation_rad,
            azimuth=azimuth_rad,
            radius=radius
        )
        
        return camera_pose.to(device)
    
    def sample_batch(self, batch_size: int, device: str = "cpu") -> torch.Tensor:
        """
        Sample a batch of camera poses.
        
        Args:
            batch_size: Number of poses to sample
            device: Target device for tensors
            
        Returns:
            Batch of camera poses [batch_size, 4, 4]
        """
        poses = []
        for _ in range(batch_size):
            pose = self.sample(device)
            poses.append(pose)
        return torch.stack(poses, dim=0)


class MultiViewDataset(AetheristDataset):
    """
    Extension of AetheristDataset for multi-view training.
    Generates multiple camera views for the same image content.
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        config: DataConfig,
        num_views: int = 2,
        split: str = "train",
        transforms_list: Optional[List] = None,
    ):
        super().__init__(data_dir, config, split, transforms_list)
        self.num_views = num_views
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get multi-view dataset item.
        
        Returns:
            Dictionary containing:
            - "images": List of transformed image tensors [num_views, 3, H, W]
            - "camera_poses": Batch of camera poses [num_views, 4, 4]
            - "path": Image path
            - "index": Sample index
        """
        # Load base image
        image_path = self.image_paths[idx]
        try:
            base_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image {image_path}: {e}")
            base_image = Image.new("RGB", (self.config.image_size, self.config.image_size))
        
        # Generate multiple views with different transforms/poses
        images = []
        camera_poses = []
        
        for view_idx in range(self.num_views):
            # Apply transforms (each view gets different augmentation)
            image_tensor = self.transforms(base_image)
            images.append(image_tensor)
            
            # Sample camera pose for this view
            camera_pose = self.camera_sampler.sample()
            camera_poses.append(camera_pose)
        
        return {
            "images": torch.stack(images, dim=0),
            "camera_poses": torch.stack(camera_poses, dim=0),
            "path": str(image_path),
            "index": idx,
        }


class SyntheticDataset(Dataset):
    """
    Synthetic dataset for testing and validation.
    Generates procedural content for model testing.
    """
    
    def __init__(
        self,
        size: int = 1000,
        image_size: int = 256,
        num_views: int = 1,
        device: str = "cpu",
    ):
        self.size = size
        self.image_size = image_size
        self.num_views = num_views
        self.device = device
        
        # Create synthetic camera sampler
        from ..config.config_manager import DataConfig
        config = DataConfig()
        config.image_size = image_size
        self.camera_sampler = CameraPoseSampler(config)
    
    def __len__(self) -> int:
        return self.size
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate synthetic data item."""
        # Generate synthetic image (colorful noise pattern)
        image = self._generate_synthetic_image(idx)
        
        if self.num_views == 1:
            camera_pose = self.camera_sampler.sample(self.device)
            return {
                "image": image,
                "camera_pose": camera_pose,
                "index": idx,
            }
        else:
            images = image.unsqueeze(0).repeat(self.num_views, 1, 1, 1)
            camera_poses = self.camera_sampler.sample_batch(self.num_views, self.device)
            return {
                "images": images,
                "camera_poses": camera_poses,
                "index": idx,
            }
    
    def _generate_synthetic_image(self, idx: int) -> torch.Tensor:
        """Generate a synthetic image pattern."""
        # Create deterministic but varied patterns
        torch.manual_seed(idx)
        
        # Generate base pattern
        image = torch.randn(3, self.image_size, self.image_size)
        
        # Add some structure
        x = torch.linspace(-1, 1, self.image_size)
        y = torch.linspace(-1, 1, self.image_size)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # Create patterns based on index
        pattern_type = idx % 4
        if pattern_type == 0:
            # Radial pattern
            pattern = torch.sin(5 * torch.sqrt(xx**2 + yy**2))
        elif pattern_type == 1:
            # Checkerboard
            pattern = torch.sin(10 * xx) * torch.sin(10 * yy)
        elif pattern_type == 2:
            # Spiral
            pattern = torch.sin(10 * (xx * torch.cos(yy) - yy * torch.sin(xx)))
        else:
            # Diagonal stripes
            pattern = torch.sin(10 * (xx + yy))
        
        # Combine with noise
        for c in range(3):
            image[c] = 0.7 * pattern + 0.3 * image[c]
        
        # Normalize to [-1, 1]
        image = torch.tanh(image)
        
        return image


def create_dataloaders(
    config: DataConfig,
    data_dir: Optional[str] = None,
    num_views: int = 1,
    use_synthetic: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create train and validation dataloaders.
    
    Args:
        config: Data configuration
        data_dir: Directory containing images (if None, uses config)
        num_views: Number of views for multi-view training
        use_synthetic: Whether to use synthetic data
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if data_dir is None:
        data_dir = config.dataset_path
    
    if use_synthetic:
        # Create synthetic datasets
        train_dataset = SyntheticDataset(
            size=1000,
            image_size=config.image_size,
            num_views=num_views,
        )
        val_dataset = SyntheticDataset(
            size=200,
            image_size=config.image_size,
            num_views=num_views,
        )
    else:
        # Create real datasets
        dataset_class = MultiViewDataset if num_views > 1 else AetheristDataset
        
        train_dataset = dataset_class(
            data_dir=data_dir,
            config=config,
            split="train",
            num_views=num_views if num_views > 1 else None,
        )
        
        # Create validation dataset if split is configured
        if config.val_split > 0:
            val_dataset = dataset_class(
                data_dir=data_dir,
                config=config,
                split="val",
                num_views=num_views if num_views > 1 else None,
            )
        else:
            val_dataset = None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last,
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,  # Don't shuffle validation
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False,  # Don't drop last for validation
        )
    
    logger.info(f"Created train loader with {len(train_loader)} batches")
    if val_loader is not None:
        logger.info(f"Created validation loader with {len(val_loader)} batches")
    
    return train_loader, val_loader


class DatasetAnalyzer:
    """
    Utility class for analyzing dataset properties.
    Provides insights into data distribution and quality.
    """
    
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
    
    def analyze_images(self, num_samples: int = 100) -> Dict:
        """Analyze image properties in the dataset."""
        logger.info(f"Analyzing {num_samples} samples from dataset...")
        
        stats = {
            "mean": torch.zeros(3),
            "std": torch.zeros(3),
            "min_val": float('inf'),
            "max_val": float('-inf'),
            "shapes": [],
        }
        
        for i in range(min(num_samples, len(self.dataset))):
            sample = self.dataset[i]
            
            if "images" in sample:
                # Multi-view dataset
                image = sample["images"][0]  # Use first view
            else:
                image = sample["image"]
            
            # Accumulate statistics
            stats["mean"] += image.mean(dim=[1, 2])
            stats["std"] += image.std(dim=[1, 2])
            stats["min_val"] = min(stats["min_val"], image.min().item())
            stats["max_val"] = max(stats["max_val"], image.max().item())
            stats["shapes"].append(image.shape)
        
        # Compute averages
        stats["mean"] /= num_samples
        stats["std"] /= num_samples
        stats["unique_shapes"] = list(set(stats["shapes"]))
        
        return stats
    
    def analyze_camera_poses(self, num_samples: int = 100) -> Dict:
        """Analyze camera pose distribution."""
        logger.info(f"Analyzing camera poses from {num_samples} samples...")
        
        poses = []
        for i in range(min(num_samples, len(self.dataset))):
            sample = self.dataset[i]
            
            if "camera_poses" in sample:
                # Multi-view dataset
                pose = sample["camera_poses"][0]  # Use first pose
            else:
                pose = sample["camera_pose"]
            
            poses.append(pose)
        
        poses = torch.stack(poses, dim=0)  # [num_samples, 4, 4]
        
        # Extract camera positions (translation part)
        positions = poses[:, :3, 3]  # [num_samples, 3]
        
        # Compute statistics
        stats = {
            "mean_position": positions.mean(dim=0),
            "std_position": positions.std(dim=0),
            "min_position": positions.min(dim=0)[0],
            "max_position": positions.max(dim=0)[0],
            "distance_from_origin": torch.norm(positions, dim=1),
        }
        
        stats["mean_distance"] = stats["distance_from_origin"].mean()
        stats["std_distance"] = stats["distance_from_origin"].std()
        
        return stats
    
    def print_analysis(self):
        """Print comprehensive dataset analysis."""
        print("\n" + "="*50)
        print("DATASET ANALYSIS")
        print("="*50)
        
        print(f"Dataset size: {len(self.dataset)}")
        
        # Image analysis
        image_stats = self.analyze_images()
        print(f"\nImage Statistics:")
        print(f"  Mean: {image_stats['mean']}")
        print(f"  Std:  {image_stats['std']}")
        print(f"  Range: [{image_stats['min_val']:.3f}, {image_stats['max_val']:.3f}]")
        print(f"  Shapes: {image_stats['unique_shapes']}")
        
        # Camera analysis
        camera_stats = self.analyze_camera_poses()
        print(f"\nCamera Pose Statistics:")
        print(f"  Mean position: {camera_stats['mean_position']}")
        print(f"  Std position:  {camera_stats['std_position']}")
        print(f"  Distance from origin: {camera_stats['mean_distance']:.3f} ± {camera_stats['std_distance']:.3f}")
        
        print("="*50 + "\n")