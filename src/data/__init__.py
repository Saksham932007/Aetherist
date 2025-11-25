"""
Aetherist Data Module
====================

This module provides comprehensive data loading and processing utilities for Aetherist,
including datasets, data loaders, camera pose sampling, and data analysis tools.

Classes:
    AetheristDataset: Main dataset class for image loading and preprocessing
    MultiViewDataset: Multi-view dataset for 3D consistency training
    SyntheticDataset: Procedural dataset for testing and validation
    CameraPoseSampler: Camera pose sampling for 3D-aware training
    DatasetAnalyzer: Analysis tools for dataset properties

Functions:
    create_dataloaders: Convenience function for creating train/val loaders

Examples:
    Basic dataset usage:
    
    >>> from src.data import AetheristDataset, create_dataloaders
    >>> from src.config import DataConfig
    >>> 
    >>> config = DataConfig()
    >>> dataset = AetheristDataset("data/", config)
    >>> train_loader, val_loader = create_dataloaders(config)
    
    Multi-view training:
    
    >>> from src.data import MultiViewDataset
    >>> dataset = MultiViewDataset("data/", config, num_views=4)
    
    Synthetic data for testing:
    
    >>> from src.data import SyntheticDataset
    >>> synthetic = SyntheticDataset(size=1000, image_size=256)
    
    Dataset analysis:
    
    >>> from src.data import DatasetAnalyzer
    >>> analyzer = DatasetAnalyzer(dataset)
    >>> analyzer.print_analysis()
"""

from .aetherist_dataset import (
    AetheristDataset,
    MultiViewDataset,
    SyntheticDataset,
    CameraPoseSampler,
    DatasetAnalyzer,
    create_dataloaders,
)

__all__ = [
    "AetheristDataset",
    "MultiViewDataset", 
    "SyntheticDataset",
    "CameraPoseSampler",
    "DatasetAnalyzer",
    "create_dataloaders",
]

__version__ = "1.0.0"