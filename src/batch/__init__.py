"""Batch processing utilities for Aetherist.

Provides high-performance batch processing capabilities for
large-scale image generation and data processing tasks.
"""

from .batch_processor import (
    BatchProcessor,
    BatchJob,
    BatchConfig,
    dummy_processor,
    image_resize_processor,
    tensor_computation_processor
)

__all__ = [
    "BatchProcessor",
    "BatchJob", 
    "BatchConfig",
    "dummy_processor",
    "image_resize_processor",
    "tensor_computation_processor"
]

__version__ = "1.0.0"
