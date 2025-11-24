#!/usr/bin/env python3
"""
Data preprocessing utilities for Aetherist.
Scripts for preparing FFHQ and other datasets for training.
"""

import os
import argparse
import shutil
from pathlib import Path
from typing import Tuple, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

import torch
from PIL import Image, ImageOps
import numpy as np
from tqdm import tqdm


def resize_and_center_crop(
    input_path: Path,
    output_path: Path,
    target_size: int = 256,
    quality: int = 95,
) -> bool:
    """
    Resize and center crop an image to target size.
    
    Args:
        input_path: Path to input image
        output_path: Path to save processed image
        target_size: Target square size
        quality: JPEG quality (if saving as JPEG)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        with Image.open(input_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Get image dimensions
            width, height = img.size
            
            # Calculate crop dimensions for center crop to square
            if width > height:
                # Wider than tall - crop width
                crop_size = height
                left = (width - height) // 2
                top = 0
                right = left + height
                bottom = height
            else:
                # Taller than wide - crop height
                crop_size = width
                left = 0
                top = (height - width) // 2
                right = width
                bottom = top + width
            
            # Center crop to square
            img = img.crop((left, top, right, bottom))
            
            # Resize to target size
            img = img.resize((target_size, target_size), Image.LANCZOS)
            
            # Create output directory if needed
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with appropriate quality
            if output_path.suffix.lower() in ['.jpg', '.jpeg']:
                img.save(output_path, 'JPEG', quality=quality, optimize=True)
            else:
                img.save(output_path)
            
            return True
    
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
        return False


def process_ffhq_dataset(
    input_dir: str,
    output_dir: str,
    target_size: int = 256,
    num_workers: int = 8,
    max_images: Optional[int] = None,
) -> None:
    """
    Process FFHQ dataset with resizing and organization.
    
    Args:
        input_dir: Input directory containing FFHQ images
        output_dir: Output directory for processed images
        target_size: Target image size (square)
        num_workers: Number of parallel workers
        max_images: Maximum number of images to process (None for all)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    image_files = sorted(image_files)
    
    if max_images:
        image_files = image_files[:max_images]
    
    print(f"Found {len(image_files)} images to process")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process images in parallel
    success_count = 0
    
    def process_single_image(input_file: Path) -> bool:
        output_file = output_path / f"{input_file.stem}.jpg"
        return resize_and_center_crop(input_file, output_file, target_size)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_image, img): img for img in image_files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
            if future.result():
                success_count += 1
    
    print(f"Successfully processed {success_count}/{len(image_files)} images")
    
    # Create dataset info file
    dataset_info = {
        'name': 'FFHQ',
        'total_images': success_count,
        'resolution': target_size,
        'format': 'jpg',
        'source_dir': str(input_path),
        'processed_dir': str(output_path),
    }
    
    with open(output_path / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"Dataset info saved to {output_path / 'dataset_info.json'}")


def create_train_val_split(
    dataset_dir: str,
    train_ratio: float = 0.9,
    seed: int = 42,
) -> None:
    """
    Split dataset into train/validation sets.
    
    Args:
        dataset_dir: Directory containing processed images
        train_ratio: Ratio of images to use for training
        seed: Random seed for reproducible splits
    """
    import random
    
    dataset_path = Path(dataset_dir)
    
    # Find all images
    image_files = list(dataset_path.glob("*.jpg"))
    image_files.extend(dataset_path.glob("*.png"))
    image_files = sorted(image_files)
    
    if not image_files:
        raise ValueError(f"No images found in {dataset_dir}")
    
    # Set random seed
    random.seed(seed)
    random.shuffle(image_files)
    
    # Split
    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    # Create directories
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Move files
    print(f"Moving {len(train_files)} files to train directory...")
    for file_path in tqdm(train_files):
        shutil.move(str(file_path), str(train_dir / file_path.name))
    
    print(f"Moving {len(val_files)} files to val directory...")
    for file_path in tqdm(val_files):
        shutil.move(str(file_path), str(val_dir / file_path.name))
    
    print(f"Split complete: {len(train_files)} train, {len(val_files)} val")


def generate_camera_metadata(
    dataset_dir: str,
    output_file: str = "camera_metadata.json",
    seed: int = 42,
) -> None:
    """
    Generate random camera poses for images in the dataset.
    
    Args:
        dataset_dir: Directory containing images
        output_file: Output metadata file name
        seed: Random seed for reproducible metadata
    """
    import random
    
    dataset_path = Path(dataset_dir)
    
    # Find all images
    image_files = []
    for subdir in [dataset_path, dataset_path / "train", dataset_path / "val"]:
        if subdir.exists():
            image_files.extend(subdir.glob("*.jpg"))
            image_files.extend(subdir.glob("*.png"))
    
    image_files = sorted(image_files)
    
    if not image_files:
        raise ValueError(f"No images found in {dataset_dir}")
    
    print(f"Generating camera metadata for {len(image_files)} images...")
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    metadata = {}
    
    for img_file in tqdm(image_files):
        # Generate random camera parameters
        elevation = np.random.uniform(-15, 15)  # degrees
        azimuth = np.random.uniform(0, 360)     # degrees  
        radius = np.random.uniform(1.0, 1.5)   # distance
        fov = np.random.uniform(45, 55)         # field of view
        
        # Create relative path from dataset root
        rel_path = str(img_file.relative_to(dataset_path))
        
        metadata[rel_path] = {
            'camera': {
                'elevation': float(elevation),
                'azimuth': float(azimuth),
                'radius': float(radius),
                'fov': float(fov),
            },
            'image_path': rel_path,
        }
    
    # Save metadata
    output_path = dataset_path / output_file
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Camera metadata saved to {output_path}")


def verify_dataset(dataset_dir: str) -> Dict[str, any]:
    """
    Verify dataset integrity and compute statistics.
    
    Args:
        dataset_dir: Directory containing processed dataset
        
    Returns:
        Dictionary with verification results
    """
    dataset_path = Path(dataset_dir)
    
    # Count images
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"
    
    stats = {
        'total_images': 0,
        'train_images': 0,
        'val_images': 0,
        'corrupted_images': [],
        'resolution_stats': {},
    }
    
    # Check train directory
    if train_dir.exists():
        train_images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
        stats['train_images'] = len(train_images)
        
        print(f"Verifying {len(train_images)} training images...")
        for img_path in tqdm(train_images):
            try:
                with Image.open(img_path) as img:
                    size = img.size
                    resolution = f"{size[0]}x{size[1]}"
                    stats['resolution_stats'][resolution] = stats['resolution_stats'].get(resolution, 0) + 1
            except Exception as e:
                stats['corrupted_images'].append(str(img_path))
                print(f"Corrupted image: {img_path}")
    
    # Check val directory  
    if val_dir.exists():
        val_images = list(val_dir.glob("*.jpg")) + list(val_dir.glob("*.png"))
        stats['val_images'] = len(val_images)
        
        print(f"Verifying {len(val_images)} validation images...")
        for img_path in tqdm(val_images):
            try:
                with Image.open(img_path) as img:
                    size = img.size
                    resolution = f"{size[0]}x{size[1]}"
                    stats['resolution_stats'][resolution] = stats['resolution_stats'].get(resolution, 0) + 1
            except Exception as e:
                stats['corrupted_images'].append(str(img_path))
                print(f"Corrupted image: {img_path}")
    
    stats['total_images'] = stats['train_images'] + stats['val_images']
    
    print(f"\nDataset Verification Results:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Train images: {stats['train_images']}")
    print(f"  Val images: {stats['val_images']}")
    print(f"  Corrupted images: {len(stats['corrupted_images'])}")
    print(f"  Resolutions: {stats['resolution_stats']}")
    
    return stats


def main():
    """Command-line interface for data preprocessing."""
    parser = argparse.ArgumentParser(description="Data preprocessing for Aetherist")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process FFHQ command
    ffhq_parser = subparsers.add_parser('process-ffhq', help='Process FFHQ dataset')
    ffhq_parser.add_argument('input_dir', help='Input directory with FFHQ images')
    ffhq_parser.add_argument('output_dir', help='Output directory for processed images')
    ffhq_parser.add_argument('--size', type=int, default=256, help='Target image size')
    ffhq_parser.add_argument('--workers', type=int, default=8, help='Number of workers')
    ffhq_parser.add_argument('--max-images', type=int, help='Maximum images to process')
    
    # Split dataset command
    split_parser = subparsers.add_parser('split', help='Create train/val split')
    split_parser.add_argument('dataset_dir', help='Dataset directory')
    split_parser.add_argument('--ratio', type=float, default=0.9, help='Train ratio')
    split_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Generate metadata command
    meta_parser = subparsers.add_parser('generate-metadata', help='Generate camera metadata')
    meta_parser.add_argument('dataset_dir', help='Dataset directory')
    meta_parser.add_argument('--output', default='camera_metadata.json', help='Output file')
    meta_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Verify dataset command
    verify_parser = subparsers.add_parser('verify', help='Verify dataset integrity')
    verify_parser.add_argument('dataset_dir', help='Dataset directory')
    
    args = parser.parse_args()
    
    if args.command == 'process-ffhq':
        process_ffhq_dataset(
            args.input_dir,
            args.output_dir,
            args.size,
            args.workers,
            args.max_images
        )
    elif args.command == 'split':
        create_train_val_split(args.dataset_dir, args.ratio, args.seed)
    elif args.command == 'generate-metadata':
        generate_camera_metadata(args.dataset_dir, args.output, args.seed)
    elif args.command == 'verify':
        verify_dataset(args.dataset_dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()