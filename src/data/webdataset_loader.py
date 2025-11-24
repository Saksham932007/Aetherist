"""
WebDataset integration for Aetherist.
Provides high-performance streaming data loading using WebDataset format.
"""

import io
import json
from typing import Dict, Any, Optional, Callable, Iterator, Tuple
from pathlib import Path

import torch
from torch.utils.data import DataLoader, IterableDataset
import webdataset as wds
from PIL import Image
import numpy as np


class WebDatasetLoader:
    """
    High-performance data loader using WebDataset format.
    
    WebDataset stores data in TAR files with structured naming:
    - image.jpg: The actual image
    - metadata.json: Optional metadata (camera pose, etc.)
    - class.txt: Optional class label
    
    Example TAR structure:
    ```
    sample000.jpg
    sample000.json
    sample001.jpg 
    sample001.json
    ...
    ```
    """
    
    def __init__(
        self,
        urls: str,
        resolution: int = 256,
        batch_size: int = 32,
        shuffle_buffer: int = 10000,
        num_workers: int = 4,
        cache_size: int = 100,
        transform: Optional[Callable] = None,
        normalize: bool = True,
    ):
        """
        Initialize WebDataset loader.
        
        Args:
            urls: URL pattern for TAR files (e.g., "path/to/data-{000..099}.tar")
            resolution: Target image resolution
            batch_size: Batch size
            shuffle_buffer: Size of shuffle buffer for randomization
            num_workers: Number of worker processes
            cache_size: Number of samples to cache for performance
            transform: Custom transformation function
            normalize: Whether to normalize images to [-1, 1]
        """
        self.urls = urls
        self.resolution = resolution
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer
        self.num_workers = num_workers
        self.cache_size = cache_size
        self.normalize = normalize
        
        if transform is None:
            self.transform = self._default_transform()
        else:
            self.transform = transform
    
    def _default_transform(self) -> Callable:
        """Create default image transformation."""
        def transform_fn(sample):
            # Decode image
            if 'jpg' in sample:
                img_data = sample['jpg']
            elif 'png' in sample:
                img_data = sample['png']
            elif 'webp' in sample:
                img_data = sample['webp']
            else:
                raise ValueError(f"No supported image format found in sample keys: {sample.keys()}")
            
            # Load image from bytes
            img = Image.open(io.BytesIO(img_data)).convert('RGB')
            
            # Resize to square
            img = img.resize((self.resolution, self.resolution), Image.LANCZOS)
            
            # Convert to tensor
            img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
            img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
            
            # Normalize to [-1, 1] if requested
            if self.normalize:
                img_tensor = img_tensor * 2.0 - 1.0
            
            # Parse metadata if available
            metadata = {}
            if 'json' in sample:
                try:
                    metadata = json.loads(sample['json'].decode('utf-8'))
                except Exception as e:
                    print(f"Error parsing metadata: {e}")
            
            return {
                'image': img_tensor,
                'metadata': metadata,
                '__key__': sample.get('__key__', ''),
            }
        
        return transform_fn
    
    def create_dataset(self, infinite: bool = True) -> IterableDataset:
        """
        Create the WebDataset.
        
        Args:
            infinite: Whether to create an infinite dataset
            
        Returns:
            WebDataset instance
        """
        dataset = (
            wds.WebDataset(self.urls)
            .shuffle(self.shuffle_buffer)
            .decode("pilrgb")  # Automatically decode images
            .map(self.transform)
        )
        
        if infinite:
            dataset = dataset.repeat()
        
        return dataset
    
    def create_dataloader(
        self,
        infinite: bool = True,
        drop_last: bool = True,
    ) -> DataLoader:
        """
        Create DataLoader with WebDataset.
        
        Args:
            infinite: Whether to create an infinite dataset
            drop_last: Whether to drop the last incomplete batch
            
        Returns:
            DataLoader instance
        """
        dataset = self.create_dataset(infinite=infinite)
        
        # Batch the dataset
        dataset = dataset.batched(
            self.batch_size,
            partial=not drop_last,
            collation_fn=self._collate_webdataset
        )
        
        return DataLoader(
            dataset,
            batch_size=None,  # Batching handled by WebDataset
            num_workers=self.num_workers,
            pin_memory=True,
            prefetch_factor=2 if self.num_workers > 0 else None,
        )
    
    def _collate_webdataset(self, batch):
        """Custom collate function for WebDataset batches."""
        if not batch:
            return {}
        
        # Stack images
        images = torch.stack([item['image'] for item in batch])
        
        # Collect metadata
        metadata = [item['metadata'] for item in batch]
        keys = [item['__key__'] for item in batch]
        
        return {
            'image': images,
            'metadata': metadata,
            'keys': keys,
        }


class CameraAwareWebDataset:
    """
    WebDataset loader that includes camera pose information.
    Expects metadata with camera parameters for 3D-aware training.
    """
    
    def __init__(
        self,
        urls: str,
        resolution: int = 256,
        batch_size: int = 32,
        **kwargs
    ):
        """
        Initialize camera-aware WebDataset loader.
        
        Args:
            urls: URL pattern for TAR files
            resolution: Target image resolution
            batch_size: Batch size
            **kwargs: Additional arguments passed to WebDatasetLoader
        """
        self.base_loader = WebDatasetLoader(
            urls=urls,
            resolution=resolution,
            batch_size=batch_size,
            transform=self._camera_transform(),
            **kwargs
        )
    
    def _camera_transform(self) -> Callable:
        """Transform that extracts camera parameters."""
        base_transform = self.base_loader._default_transform()
        
        def camera_transform_fn(sample):
            # Apply base transform
            transformed = base_transform(sample)
            
            # Extract camera parameters from metadata
            metadata = transformed['metadata']
            
            # Default camera parameters if not available
            camera_params = {
                'elevation': 0.0,
                'azimuth': 0.0,
                'radius': 1.2,
                'fov': 50.0,
            }
            
            # Update with metadata if available
            if 'camera' in metadata:
                camera_params.update(metadata['camera'])
            
            # Convert to tensor
            camera_tensor = torch.tensor([
                camera_params['elevation'],
                camera_params['azimuth'], 
                camera_params['radius'],
            ], dtype=torch.float32)
            
            transformed['camera_params'] = camera_tensor
            transformed['fov'] = camera_params['fov']
            
            return transformed
        
        return camera_transform_fn
    
    def create_dataloader(self, **kwargs) -> DataLoader:
        """Create DataLoader with camera-aware batching."""
        return self.base_loader.create_dataloader(**kwargs)


def convert_dataset_to_webdataset(
    source_dir: str,
    output_pattern: str,
    samples_per_tar: int = 1000,
    image_format: str = "jpg",
    quality: int = 95,
) -> None:
    """
    Convert a directory of images to WebDataset format.
    
    Args:
        source_dir: Directory containing source images
        output_pattern: Output TAR file pattern (e.g., "data-%06d.tar")
        samples_per_tar: Number of samples per TAR file
        image_format: Output image format ('jpg', 'png', 'webp')
        quality: Image quality for lossy formats
    """
    from pathlib import Path
    import tarfile
    
    source_path = Path(source_dir)
    image_files = []
    
    # Find all images
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
        image_files.extend(source_path.glob(f"*{ext}"))
        image_files.extend(source_path.glob(f"*{ext.upper()}"))
    
    image_files.sort()
    
    print(f"Converting {len(image_files)} images to WebDataset format...")
    
    tar_index = 0
    sample_index = 0
    current_tar = None
    
    for i, image_path in enumerate(image_files):
        # Open new TAR file if needed
        if i % samples_per_tar == 0:
            if current_tar:
                current_tar.close()
            
            tar_path = output_pattern % tar_index
            current_tar = tarfile.open(tar_path, 'w')
            print(f"Creating {tar_path}")
            tar_index += 1
        
        try:
            # Load and process image
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                
                # Save processed image to bytes
                img_bytes = io.BytesIO()
                save_kwargs = {}
                if image_format.lower() in ['jpg', 'jpeg']:
                    save_kwargs['quality'] = quality
                    save_kwargs['optimize'] = True
                
                img.save(img_bytes, format=image_format.upper(), **save_kwargs)
                img_data = img_bytes.getvalue()
            
            # Create sample key
            sample_key = f"{sample_index:06d}"
            
            # Add image to TAR
            img_info = tarfile.TarInfo(name=f"{sample_key}.{image_format}")
            img_info.size = len(img_data)
            current_tar.addfile(img_info, io.BytesIO(img_data))
            
            # Add basic metadata
            metadata = {
                'original_path': str(image_path),
                'index': sample_index,
            }
            metadata_json = json.dumps(metadata).encode('utf-8')
            
            meta_info = tarfile.TarInfo(name=f"{sample_key}.json")
            meta_info.size = len(metadata_json)
            current_tar.addfile(meta_info, io.BytesIO(metadata_json))
            
            sample_index += 1
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_files)} images")
        
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    if current_tar:
        current_tar.close()
    
    print(f"Conversion complete! Created {tar_index} TAR files with {sample_index} samples.")


def benchmark_dataloader(
    dataloader: DataLoader,
    num_batches: int = 100,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, float]:
    """
    Benchmark dataloader performance.
    
    Args:
        dataloader: DataLoader to benchmark
        num_batches: Number of batches to process
        device: Device to transfer data to
        
    Returns:
        Performance statistics
    """
    import time
    
    print(f"Benchmarking dataloader for {num_batches} batches...")
    
    times = []
    data_sizes = []
    
    start_time = time.time()
    
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        
        batch_start = time.time()
        
        # Transfer to device
        if 'image' in batch:
            batch['image'] = batch['image'].to(device, non_blocking=True)
        
        batch_time = time.time() - batch_start
        times.append(batch_time)
        
        if 'image' in batch:
            data_sizes.append(batch['image'].numel() * 4)  # Assuming float32
    
    total_time = time.time() - start_time
    
    stats = {
        'total_time': total_time,
        'avg_batch_time': np.mean(times),
        'std_batch_time': np.std(times),
        'min_batch_time': np.min(times),
        'max_batch_time': np.max(times),
        'throughput_samples_per_sec': (num_batches * len(batch['image'])) / total_time if 'image' in batch else 0,
        'throughput_mb_per_sec': np.sum(data_sizes) / (1024 * 1024) / total_time,
    }
    
    print(f"Benchmark Results:")
    print(f"  Total time: {stats['total_time']:.2f}s")
    print(f"  Avg batch time: {stats['avg_batch_time']:.4f}s")
    print(f"  Throughput: {stats['throughput_samples_per_sec']:.1f} samples/s")
    print(f"  Data throughput: {stats['throughput_mb_per_sec']:.1f} MB/s")
    
    return stats