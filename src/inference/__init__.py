"""
Aetherist Inference Module
=========================

This module provides inference utilities for the Aetherist GAN, including:
- Model loading and checkpoint management
- Image generation from latent codes
- Latent interpolation and exploration
- Batch processing for large-scale generation
- Web interface for interactive use

Classes:
    AetheristInferencePipeline: Main inference pipeline
    BatchInferencePipeline: Optimized batch processing
    
Examples:
    Basic usage:
    
    >>> from src.inference import AetheristInferencePipeline
    >>> pipeline = AetheristInferencePipeline("model.pth")
    >>> result = pipeline.generate(num_samples=4)
    >>> pipeline.save_images(result["images"], "output/")
    
    Latent interpolation:
    
    >>> latent1 = torch.randn(256)
    >>> latent2 = torch.randn(256)
    >>> result = pipeline.interpolate(latent1, latent2, steps=10)
    
    Batch generation:
    
    >>> from src.inference import BatchInferencePipeline
    >>> batch_pipeline = BatchInferencePipeline(pipeline)
    >>> batch_pipeline.generate_large_batch(100, "output/")
"""

from .inference_pipeline import (
    AetheristInferencePipeline,
    BatchInferencePipeline,
)

__all__ = [
    "AetheristInferencePipeline",
    "BatchInferencePipeline",
]

__version__ = "1.0.0"