"""
Dataset classes for Aetherist.
Provides data loading utilities for training the Hybrid 3D-ViT-GAN.
"""

import os
import random
from typing import Tuple, Optional, List, Dict, Any, Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np


class ImageDataset(Dataset):
    """
    Base dataset class for loading images from a directory structure.
    
    Supports common image formats and provides data augmentation.
    Expected directory structure:
    ```
    dataset_path/
    ├── image1.jpg
    ├── image2.png
    ├── ...
    └── imageN.jpg
    ```
    """
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    def __init__(
        self,
        dataset_path: str,
        resolution: int = 256,
        transform: Optional[Callable] = None,
        augmentation_prob: float = 0.5,
        center_crop: bool = True,
        normalize: bool = True,
        cache_images: bool = False,
    ):
        """
        Initialize the image dataset.
        
        Args:
            dataset_path: Path to the dataset directory
            resolution: Target image resolution (square images)
            transform: Custom transformation function
            augmentation_prob: Probability of applying augmentations
            center_crop: Whether to center crop images before resizing
            normalize: Whether to normalize images to [-1, 1] range
            cache_images: Whether to cache loaded images in memory (use with caution)
        """
        self.dataset_path = Path(dataset_path)
        self.resolution = resolution
        self.augmentation_prob = augmentation_prob
        self.center_crop = center_crop
        self.normalize = normalize
        self.cache_images = cache_images
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
        
        # Find all image files
        self.image_paths = self._find_image_files()
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {dataset_path}")
        
        print(f"Found {len(self.image_paths)} images in {dataset_path}")
        
        # Setup transforms
        if transform is None:
            self.transform = self._default_transform()
        else:
            self.transform = transform
        
        # Image cache
        self._image_cache: Dict[int, torch.Tensor] = {}
    
    def _find_image_files(self) -> List[Path]:
        """Find all image files in the dataset directory."""
        image_files = []
        
        for ext in self.SUPPORTED_EXTENSIONS:
            image_files.extend(self.dataset_path.glob(f"*{ext}"))
            image_files.extend(self.dataset_path.glob(f"*{ext.upper()}"))
        
        # Sort for consistent ordering
        image_files.sort()
        return image_files
    
    def _default_transform(self) -> transforms.Compose:
        """Create default image transformation pipeline."""
        transform_list = []
        
        # Center crop to square if requested
        if self.center_crop:
            transform_list.append(transforms.CenterCrop(min))
        
        # Resize to target resolution
        transform_list.extend([
            transforms.Resize((self.resolution, self.resolution), antialias=True),
            transforms.ToTensor(),
        ])
        
        # Normalize to [-1, 1] if requested
        if self.normalize:
            transform_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
        
        return transforms.Compose(transform_list)
    
    def _apply_augmentation(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to the image.
        
        Args:
            image: Input image tensor (C, H, W)
            
        Returns:
            Augmented image tensor
        """
        if random.random() > self.augmentation_prob:
            return image
        
        # Random horizontal flip
        if random.random() < 0.5:
            image = TF.hflip(image)
        
        # Random rotation (small angles)
        if random.random() < 0.3:
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle)
        
        # Random color jitter
        if random.random() < 0.3:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            saturation = random.uniform(0.8, 1.2)
            hue = random.uniform(-0.1, 0.1)
            
            image = TF.adjust_brightness(image, brightness)
            image = TF.adjust_contrast(image, contrast)
            image = TF.adjust_saturation(image, saturation)
            image = TF.adjust_hue(image, hue)
        
        return image
    
    def _load_image(self, idx: int) -> torch.Tensor:
        """Load and preprocess a single image."""
        if self.cache_images and idx in self._image_cache:
            return self._image_cache[idx]
        
        image_path = self.image_paths[idx]
        
        try:
            # Load image
            with Image.open(image_path) as img:
                # Convert to RGB if needed
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Apply base transform
                image_tensor = self.transform(img)
                
                # Apply augmentations
                image_tensor = self._apply_augmentation(image_tensor)
                
                # Cache if requested
                if self.cache_images:
                    self._image_cache[idx] = image_tensor.clone()
                
                return image_tensor
        
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a random tensor as fallback
            return torch.randn(3, self.resolution, self.resolution)
    
    def __len__(self) -> int:
        """Return the size of the dataset."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a single item from the dataset.
        
        Args:
            idx: Dataset index
            
        Returns:
            Dictionary containing:
            - 'image': Image tensor (C, H, W)
            - 'index': Dataset index
            - 'path': Image file path
        """
        if idx >= len(self.image_paths):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        image = self._load_image(idx)
        
        return {
            'image': image,
            'index': idx,
            'path': str(self.image_paths[idx]),
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        return {
            'size': len(self),
            'resolution': self.resolution,
            'dataset_path': str(self.dataset_path),
            'cached_images': len(self._image_cache),
        }


class PairedImageDataset(ImageDataset):
    """
    Dataset for loading paired images (e.g., for conditional generation).
    
    Expected directory structure:
    ```
    dataset_path/
    ├── condition/
    │   ├── image1.jpg
    │   └── ...
    └── target/
        ├── image1.jpg
        └── ...
    ```
    """
    
    def __init__(
        self,
        dataset_path: str,
        condition_subdir: str = "condition",
        target_subdir: str = "target",
        **kwargs
    ):
        """
        Initialize the paired image dataset.
        
        Args:
            dataset_path: Path to the dataset directory
            condition_subdir: Subdirectory containing condition images
            target_subdir: Subdirectory containing target images
            **kwargs: Additional arguments passed to parent class
        """
        self.condition_path = Path(dataset_path) / condition_subdir
        self.target_path = Path(dataset_path) / target_subdir
        
        if not self.condition_path.exists():
            raise FileNotFoundError(f"Condition path not found: {self.condition_path}")
        if not self.target_path.exists():
            raise FileNotFoundError(f"Target path not found: {self.target_path}")
        
        # Initialize parent with condition path
        super().__init__(str(self.condition_path), **kwargs)
        
        # Find matching target images
        self.target_paths = self._find_matching_targets()
        
        if len(self.target_paths) != len(self.image_paths):
            raise ValueError(
                f"Mismatch in number of condition ({len(self.image_paths)}) "
                f"and target ({len(self.target_paths)}) images"
            )
    
    def _find_matching_targets(self) -> List[Path]:
        """Find target images matching condition images."""
        target_paths = []
        
        for condition_path in self.image_paths:
            # Find corresponding target with same filename
            target_path = self.target_path / condition_path.name
            
            if not target_path.exists():
                raise FileNotFoundError(f"Target image not found: {target_path}")
            
            target_paths.append(target_path)
        
        return target_paths
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a paired item from the dataset.
        
        Returns:
            Dictionary containing:
            - 'condition': Condition image tensor (C, H, W)
            - 'target': Target image tensor (C, H, W)
            - 'index': Dataset index
            - 'condition_path': Condition image file path
            - 'target_path': Target image file path
        """
        if idx >= len(self.image_paths):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        # Load condition image (using parent method)
        condition_data = super().__getitem__(idx)
        condition = condition_data['image']
        
        # Load target image
        target_path = self.target_paths[idx]
        try:
            with Image.open(target_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                target = self.transform(img)
                target = self._apply_augmentation(target)
        except Exception as e:
            print(f"Error loading target image {target_path}: {e}")
            target = torch.randn(3, self.resolution, self.resolution)
        
        return {
            'condition': condition,
            'target': target,
            'index': idx,
            'condition_path': str(self.image_paths[idx]),
            'target_path': str(target_path),
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """
    Create a DataLoader with sensible defaults for training.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle: Whether to shuffle the dataset
        pin_memory: Whether to pin memory for faster GPU transfer
        drop_last: Whether to drop the last incomplete batch
        prefetch_factor: Number of batches to prefetch per worker
        
    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching dataset items.
    
    Args:
        batch: List of dataset items
        
    Returns:
        Batched dictionary
    """
    if not batch:
        return {}
    
    # Get keys from first item
    keys = batch[0].keys()
    batched = {}
    
    for key in keys:
        items = [item[key] for item in batch]
        
        if isinstance(items[0], torch.Tensor):
            batched[key] = torch.stack(items, dim=0)
        elif isinstance(items[0], (int, float)):
            batched[key] = torch.tensor(items)
        else:
            batched[key] = items  # Keep as list for non-tensor items
    
    return batched