#!/usr/bin/env python3
"""
Script to create and prepare datasets for Aetherist training.
Supports dataset conversion, validation, and analysis.
"""

import argparse
import shutil
import json
from pathlib import Path
import sys
import logging
from PIL import Image
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data import AetheristDataset, DatasetAnalyzer, SyntheticDataset
from src.config import DataConfig


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def create_synthetic_dataset(output_dir: str, size: int, image_size: int):
    """Create a synthetic dataset for testing."""
    logger = logging.getLogger(__name__)
    output_dir = Path(output_dir)
    
    logger.info(f"Creating synthetic dataset with {size} images at {output_dir}")
    
    # Create directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Create synthetic dataset
    synthetic = SyntheticDataset(size=size, image_size=image_size)
    
    # Split into train/val (80/20)
    val_size = int(size * 0.2)
    train_size = size - val_size
    
    logger.info(f"Generating {train_size} training images...")
    for i in range(train_size):
        sample = synthetic[i]
        
        # Convert tensor to PIL image
        image_tensor = sample['image']
        image_tensor = (image_tensor + 1) / 2  # Convert from [-1,1] to [0,1]
        image_tensor = torch.clamp(image_tensor, 0, 1)
        image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
        image = Image.fromarray(image_np)
        
        # Save image
        image_path = train_dir / f"synthetic_{i:05d}.png"
        image.save(image_path)
    
    logger.info(f"Generating {val_size} validation images...")
    for i in range(val_size):
        sample = synthetic[train_size + i]
        
        # Convert tensor to PIL image
        image_tensor = sample['image']
        image_tensor = (image_tensor + 1) / 2
        image_tensor = torch.clamp(image_tensor, 0, 1)
        image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype('uint8')
        image = Image.fromarray(image_np)
        
        # Save image
        image_path = val_dir / f"synthetic_{i:05d}.png"
        image.save(image_path)
    
    # Save dataset metadata
    metadata = {
        "dataset_type": "synthetic",
        "total_images": size,
        "train_images": train_size,
        "val_images": val_size,
        "image_size": image_size,
        "format": "PNG",
        "created_with": "Aetherist dataset creation script"
    }
    
    metadata_path = output_dir / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Synthetic dataset created successfully!")
    logger.info(f"  Train images: {train_size}")
    logger.info(f"  Val images: {val_size}")
    logger.info(f"  Metadata saved to: {metadata_path}")


def organize_existing_dataset(input_dir: str, output_dir: str, train_split: float = 0.8):
    """Organize an existing dataset into train/val splits."""
    logger = logging.getLogger(__name__)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    logger.info(f"Organizing dataset from {input_dir} to {output_dir}")
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_dir.rglob(f"*{ext}"))
        image_files.extend(input_dir.rglob(f"*{ext.upper()}"))
    
    if not image_files:
        raise ValueError(f"No image files found in {input_dir}")
    
    logger.info(f"Found {len(image_files)} image files")
    
    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Shuffle and split
    import random
    random.shuffle(image_files)
    
    split_idx = int(len(image_files) * train_split)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    
    logger.info(f"Copying {len(train_files)} files to train directory...")
    for i, src_file in enumerate(train_files):
        dst_file = train_dir / f"image_{i:05d}{src_file.suffix}"
        shutil.copy2(src_file, dst_file)
    
    logger.info(f"Copying {len(val_files)} files to validation directory...")
    for i, src_file in enumerate(val_files):
        dst_file = val_dir / f"image_{i:05d}{src_file.suffix}"
        shutil.copy2(src_file, dst_file)
    
    # Save metadata
    metadata = {
        "dataset_type": "organized",
        "source_directory": str(input_dir),
        "total_images": len(image_files),
        "train_images": len(train_files),
        "val_images": len(val_files),
        "train_split": train_split,
        "organized_with": "Aetherist dataset creation script"
    }
    
    metadata_path = output_dir / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Dataset organized successfully!")
    logger.info(f"  Train images: {len(train_files)}")
    logger.info(f"  Val images: {len(val_files)}")


def validate_dataset(dataset_dir: str):
    """Validate dataset structure and quality."""
    logger = logging.getLogger(__name__)
    dataset_dir = Path(dataset_dir)
    
    logger.info(f"Validating dataset at {dataset_dir}")
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    # Check directory structure
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    
    if not train_dir.exists():
        logger.warning(f"No train directory found at {train_dir}")
        return False
    
    if not val_dir.exists():
        logger.warning(f"No validation directory found at {val_dir}")
    
    # Create dataset config
    config = DataConfig()
    config.dataset_path = str(dataset_dir)
    config.image_size = 256
    config.val_split = 0.0  # We already have split directories
    
    try:
        # Test dataset loading
        dataset = AetheristDataset(
            data_dir=dataset_dir,
            config=config,
            split="train"
        )
        
        logger.info(f"✅ Dataset loaded successfully: {len(dataset)} samples")
        
        # Test sample loading
        sample = dataset[0]
        logger.info(f"✅ Sample loaded: {sample['image'].shape}")
        
        # Run analysis
        analyzer = DatasetAnalyzer(dataset)
        analyzer.print_analysis()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Dataset validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def resize_dataset(dataset_dir: str, target_size: int, output_dir: str = None):
    """Resize all images in a dataset to target size."""
    logger = logging.getLogger(__name__)
    dataset_dir = Path(dataset_dir)
    
    if output_dir is None:
        output_dir = dataset_dir.parent / f"{dataset_dir.name}_resized_{target_size}"
    else:
        output_dir = Path(output_dir)
    
    logger.info(f"Resizing dataset from {dataset_dir} to {target_size}x{target_size}")
    logger.info(f"Output directory: {output_dir}")
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    for subdir in ["train", "val", "test"]:
        src_subdir = dataset_dir / subdir
        if not src_subdir.exists():
            continue
            
        dst_subdir = output_dir / subdir
        dst_subdir.mkdir(parents=True, exist_ok=True)
        
        image_files = []
        for ext in image_extensions:
            image_files.extend(src_subdir.glob(f"*{ext}"))
            image_files.extend(src_subdir.glob(f"*{ext.upper()}"))
        
        logger.info(f"Resizing {len(image_files)} images in {subdir}...")
        
        for img_file in image_files:
            try:
                # Load and resize image
                image = Image.open(img_file).convert("RGB")
                image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
                
                # Save resized image
                dst_file = dst_subdir / img_file.name
                image.save(dst_file, "PNG", quality=95)
                
            except Exception as e:
                logger.warning(f"Failed to resize {img_file}: {e}")
    
    logger.info(f"Dataset resizing completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Create and prepare datasets for Aetherist training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Create synthetic dataset
    synthetic_parser = subparsers.add_parser(
        "create-synthetic", 
        help="Create synthetic dataset for testing"
    )
    synthetic_parser.add_argument(
        "--output-dir", "-o", 
        required=True,
        help="Output directory for synthetic dataset"
    )
    synthetic_parser.add_argument(
        "--size", "-s",
        type=int, 
        default=1000,
        help="Number of synthetic images to create"
    )
    synthetic_parser.add_argument(
        "--image-size",
        type=int, 
        default=256,
        help="Size of generated images"
    )
    
    # Organize existing dataset
    organize_parser = subparsers.add_parser(
        "organize",
        help="Organize existing images into train/val splits"
    )
    organize_parser.add_argument(
        "--input-dir", "-i",
        required=True,
        help="Input directory containing images"
    )
    organize_parser.add_argument(
        "--output-dir", "-o",
        required=True, 
        help="Output directory for organized dataset"
    )
    organize_parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Fraction of data to use for training"
    )
    
    # Validate dataset
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate dataset structure and quality"
    )
    validate_parser.add_argument(
        "--dataset-dir", "-d",
        required=True,
        help="Dataset directory to validate"
    )
    
    # Resize dataset
    resize_parser = subparsers.add_parser(
        "resize",
        help="Resize all images in dataset"
    )
    resize_parser.add_argument(
        "--dataset-dir", "-d",
        required=True,
        help="Dataset directory to resize"
    )
    resize_parser.add_argument(
        "--size", "-s",
        type=int,
        required=True,
        help="Target image size"
    )
    resize_parser.add_argument(
        "--output-dir", "-o",
        help="Output directory (default: auto-generated)"
    )
    
    # Global options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        if args.command == "create-synthetic":
            create_synthetic_dataset(
                output_dir=args.output_dir,
                size=args.size,
                image_size=args.image_size
            )
        
        elif args.command == "organize":
            organize_existing_dataset(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                train_split=args.train_split
            )
        
        elif args.command == "validate":
            success = validate_dataset(args.dataset_dir)
            if not success:
                sys.exit(1)
        
        elif args.command == "resize":
            resize_dataset(
                dataset_dir=args.dataset_dir,
                target_size=args.size,
                output_dir=args.output_dir
            )
        
        logger.info("Dataset operation completed successfully! 🎉")
        
    except Exception as e:
        logger.error(f"Dataset operation failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()