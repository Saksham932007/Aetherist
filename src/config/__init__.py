"""
Aetherist Configuration Module
=============================

This module provides comprehensive configuration management for Aetherist,
including YAML-based configuration files and dataclass structures.

Classes:
    ModelConfig: Model architecture parameters
    TrainingConfig: Training and optimization settings
    DataConfig: Dataset and data loading configuration
    InferenceConfig: Inference pipeline settings
    ExperimentConfig: Experiment tracking and output settings
    AetheristConfig: Complete configuration container
    ConfigManager: Configuration loading and management

Examples:
    Load default configuration:
    
    >>> from src.config import load_config
    >>> config = load_config()
    
    Load from YAML file:
    
    >>> config = load_config("configs/high_quality.yaml")
    
    Load with overrides:
    
    >>> config = load_config(
    ...     "configs/base.yaml",
    ...     overrides=["training.batch_size=16", "model.latent_dim=512"]
    ... )
    
    Create experiment configuration:
    
    >>> from src.config import ConfigManager
    >>> manager = ConfigManager()
    >>> exp_config = manager.create_experiment_config(
    ...     config, "my_experiment"
    ... )
"""

from .config_manager import (
    ModelConfig,
    TrainingConfig,
    DataConfig,
    InferenceConfig,
    ExperimentConfig,
    AetheristConfig,
    ConfigManager,
    config_manager,
    load_config,
    save_config,
)

__all__ = [
    "ModelConfig",
    "TrainingConfig", 
    "DataConfig",
    "InferenceConfig",
    "ExperimentConfig",
    "AetheristConfig",
    "ConfigManager",
    "config_manager",
    "load_config",
    "save_config",
]

__version__ = "1.0.0"