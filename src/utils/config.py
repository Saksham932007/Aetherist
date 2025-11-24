"""
Configuration utilities for Aetherist.
Provides Hydra configuration loading and validation.
"""

from typing import Any, Dict, Optional
from pathlib import Path
import os

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf


def load_config(
    config_path: Optional[str] = None,
    config_name: str = "config",
    overrides: Optional[list] = None,
) -> DictConfig:
    """
    Load configuration using Hydra.
    
    Args:
        config_path: Path to config directory. If None, uses default configs/ directory
        config_name: Name of config file (without .yaml extension)
        overrides: List of config overrides in hydra format (e.g., ["model.batch_size=8"])
        
    Returns:
        DictConfig: Loaded configuration
    """
    if config_path is None:
        # Get the project root directory
        current_dir = Path(__file__).parent.parent.parent
        config_path = str(current_dir / "configs")
    
    # Clear any existing Hydra instance
    GlobalHydra.instance().clear()
    
    # Initialize Hydra with config directory
    with initialize_config_dir(config_dir=config_path, version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    
    return cfg


def validate_config(cfg: DictConfig) -> None:
    """
    Validate configuration parameters.
    
    Args:
        cfg: Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Validate model parameters
    assert cfg.model.generator.latent_dim > 0, "Latent dimension must be positive"
    assert cfg.model.generator.vit_layers > 0, "Number of ViT layers must be positive"
    assert cfg.model.generator.triplane_resolution > 0, "Triplane resolution must be positive"
    
    # Validate training parameters
    assert cfg.training.batch_size > 0, "Batch size must be positive"
    assert cfg.training.generator_lr > 0, "Generator learning rate must be positive"
    assert cfg.training.discriminator_lr > 0, "Discriminator learning rate must be positive"
    assert 0 <= cfg.training.beta1 <= 1, "Beta1 must be in [0, 1]"
    assert 0 <= cfg.training.beta2 <= 1, "Beta2 must be in [0, 1]"
    
    # Validate data parameters
    assert cfg.data.resolution > 0, "Data resolution must be positive"
    assert cfg.data.num_workers >= 0, "Number of workers must be non-negative"
    
    # Validate camera parameters
    assert cfg.data.camera_fov > 0, "Camera FOV must be positive"
    assert cfg.data.camera_radius > 0, "Camera radius must be positive"


def print_config(cfg: DictConfig, resolve: bool = True) -> None:
    """
    Pretty print configuration.
    
    Args:
        cfg: Configuration to print
        resolve: Whether to resolve interpolations
    """
    print("Configuration:")
    print("-" * 50)
    print(OmegaConf.to_yaml(cfg, resolve=resolve))
    print("-" * 50)


def save_config(cfg: DictConfig, save_path: str) -> None:
    """
    Save configuration to file.
    
    Args:
        cfg: Configuration to save
        save_path: Path to save configuration
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        OmegaConf.save(cfg, f)