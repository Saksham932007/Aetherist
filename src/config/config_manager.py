"""
Configuration management system for Aetherist.
Handles YAML-based configurations for training and inference.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from omegaconf import OmegaConf, DictConfig
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    # Generator settings
    latent_dim: int = 256
    vit_dim: int = 256
    vit_layers: int = 8
    vit_heads: int = 8
    vit_mlp_ratio: float = 4.0
    triplane_resolution: int = 64
    triplane_channels: int = 32
    
    # Neural renderer settings
    ray_samples_coarse: int = 64
    ray_samples_fine: int = 64
    density_noise: float = 0.0
    
    # Super-resolution settings
    sr_hidden_dim: int = 128
    sr_num_layers: int = 4
    
    # Discriminator settings
    discriminator_base_channels: int = 64
    discriminator_feature_dim: int = 256
    use_multiscale_discriminator: bool = True
    use_consistency_branch: bool = True
    num_views_consistency: int = 2


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    # Basic training settings
    batch_size: int = 32
    num_epochs: int = 500
    learning_rate_g: float = 2e-4
    learning_rate_d: float = 2e-4
    beta1: float = 0.0
    beta2: float = 0.99
    weight_decay: float = 0.0
    
    # Loss weights
    lambda_adversarial: float = 1.0
    lambda_perceptual: float = 0.1
    lambda_consistency: float = 0.05
    lambda_r1: float = 10.0
    
    # Training schedule
    warmup_epochs: int = 10
    scheduler_type: str = "cosine"  # "cosine", "step", "linear"
    scheduler_gamma: float = 0.95
    scheduler_step_size: int = 50
    
    # Multi-view training
    use_multi_view: bool = True
    num_views: int = 2
    view_probability: float = 0.5
    
    # Regularization
    gradient_clip_norm: float = 1.0
    ema_decay: float = 0.999
    use_ema: bool = True
    
    # Progressive training
    progressive_training: bool = False
    progressive_epochs: List[int] = field(default_factory=lambda: [100, 200, 300])
    progressive_resolutions: List[int] = field(default_factory=lambda: [64, 128, 256])
    
    # Logging and checkpointing
    log_every: int = 50
    save_every: int = 1000
    sample_every: int = 500
    validate_every: int = 2000
    max_checkpoints: int = 5


@dataclass
class DataConfig:
    """Configuration for dataset and data loading."""
    dataset_path: str = "data/"
    image_size: int = 256
    num_workers: int = 8
    pin_memory: bool = True
    shuffle: bool = True
    drop_last: bool = True
    
    # Data augmentation
    use_augmentation: bool = True
    horizontal_flip: bool = True
    color_jitter: bool = False
    rotation_degrees: int = 0
    
    # Camera sampling
    camera_radius_range: List[float] = field(default_factory=lambda: [1.5, 2.5])
    camera_elevation_range: List[float] = field(default_factory=lambda: [-15, 45])
    camera_azimuth_range: List[float] = field(default_factory=lambda: [-180, 180])
    
    # Validation split
    val_split: float = 0.1
    val_seed: int = 42


@dataclass
class InferenceConfig:
    """Configuration for inference parameters."""
    device: str = "auto"
    half_precision: bool = True
    batch_size: int = 8
    max_batch_size: int = 16
    
    # Generation settings
    default_num_samples: int = 4
    default_resolution: int = 256
    default_seed: Optional[int] = None
    
    # Camera settings
    default_radius: float = 2.0
    default_elevation: float = 15.0
    default_azimuth: float = 0.0
    
    # Post-processing
    output_format: str = "PNG"
    save_latents: bool = False
    save_cameras: bool = False
    save_metadata: bool = True
    
    # Web interface
    web_host: str = "127.0.0.1"
    web_port: int = 7860
    web_share: bool = False


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking and output."""
    name: str = "aetherist_experiment"
    output_dir: str = "experiments/"
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "aetherist"
    wandb_entity: Optional[str] = None
    log_level: str = "INFO"
    
    # Checkpointing
    checkpoint_dir: str = "checkpoints/"
    sample_dir: str = "samples/"
    log_dir: str = "logs/"
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = False
    
    # Hardware
    device: str = "auto"
    mixed_precision: bool = True
    compile_model: bool = False


@dataclass
class AetheristConfig:
    """Complete Aetherist configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


class ConfigManager:
    """
    Configuration manager for Aetherist.
    
    Handles loading, saving, and validation of YAML configurations.
    Supports OmegaConf for advanced configuration features.
    """
    
    def __init__(self):
        self.config = None
        self._config_path = None
    
    def load_config(
        self, 
        config_path: Optional[Union[str, Path]] = None,
        overrides: Optional[List[str]] = None,
    ) -> AetheristConfig:
        """
        Load configuration from YAML file with optional overrides.
        
        Args:
            config_path: Path to YAML configuration file
            overrides: List of override strings in format "key=value"
            
        Returns:
            Complete Aetherist configuration
        """
        if config_path is None:
            # Create default configuration
            logger.info("No config file specified, using default configuration")
            config = AetheristConfig()
        else:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            logger.info(f"Loading configuration from {config_path}")
            
            # Load YAML configuration
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)
            
            # Convert to OmegaConf for advanced features
            omega_config = OmegaConf.create(yaml_config)
            
            # Apply overrides if provided
            if overrides:
                logger.info(f"Applying {len(overrides)} configuration overrides")
                for override in overrides:
                    logger.debug(f"Override: {override}")
                omega_config = OmegaConf.merge(
                    omega_config, 
                    OmegaConf.from_dotlist(overrides)
                )
            
            # Convert to dataclass
            config = self._omega_to_dataclass(omega_config)
            self._config_path = config_path
        
        # Validate configuration
        self._validate_config(config)
        
        self.config = config
        return config
    
    def save_config(
        self, 
        config: AetheristConfig, 
        save_path: Union[str, Path],
        save_format: str = "yaml"
    ) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            save_path: Path to save configuration
            save_format: Format to save ("yaml" or "json")
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        config_dict = asdict(config)
        
        if save_format.lower() == "yaml":
            with open(save_path, 'w') as f:
                yaml.safe_dump(
                    config_dict, 
                    f, 
                    default_flow_style=False,
                    indent=2,
                    sort_keys=False
                )
        elif save_format.lower() == "json":
            with open(save_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")
        
        logger.info(f"Configuration saved to {save_path}")
    
    def create_experiment_config(
        self, 
        base_config: AetheristConfig, 
        experiment_name: str,
        output_dir: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> AetheristConfig:
        """
        Create experiment-specific configuration.
        
        Args:
            base_config: Base configuration to modify
            experiment_name: Name of the experiment
            output_dir: Override output directory
            overrides: Dictionary of configuration overrides
            
        Returns:
            Experiment-specific configuration
        """
        # Deep copy the configuration
        import copy
        exp_config = copy.deepcopy(base_config)
        
        # Update experiment settings
        exp_config.experiment.name = experiment_name
        if output_dir is not None:
            exp_config.experiment.output_dir = output_dir
        
        # Create experiment subdirectories
        exp_dir = Path(exp_config.experiment.output_dir) / experiment_name
        exp_config.experiment.checkpoint_dir = str(exp_dir / "checkpoints")
        exp_config.experiment.sample_dir = str(exp_dir / "samples")
        exp_config.experiment.log_dir = str(exp_dir / "logs")
        
        # Apply overrides if provided
        if overrides:
            exp_config = self._apply_overrides(exp_config, overrides)
        
        return exp_config
    
    def _omega_to_dataclass(self, omega_config: DictConfig) -> AetheristConfig:
        """Convert OmegaConf configuration to dataclass."""
        # Convert OmegaConf to regular dict
        config_dict = OmegaConf.to_container(omega_config, resolve=True)
        
        # Create dataclass instances
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        inference_config = InferenceConfig(**config_dict.get("inference", {}))
        experiment_config = ExperimentConfig(**config_dict.get("experiment", {}))
        
        return AetheristConfig(
            model=model_config,
            training=training_config,
            data=data_config,
            inference=inference_config,
            experiment=experiment_config
        )
    
    def _apply_overrides(
        self, 
        config: AetheristConfig, 
        overrides: Dict[str, Any]
    ) -> AetheristConfig:
        """Apply dictionary overrides to configuration."""
        import copy
        new_config = copy.deepcopy(config)
        
        for key, value in overrides.items():
            if "." in key:
                # Handle nested keys like "model.latent_dim"
                parts = key.split(".")
                obj = new_config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
            else:
                # Handle top-level keys
                if hasattr(new_config, key):
                    setattr(new_config, key, value)
        
        return new_config
    
    def _validate_config(self, config: AetheristConfig) -> None:
        """Validate configuration parameters."""
        # Model validation
        assert config.model.latent_dim > 0, "Latent dimension must be positive"
        assert config.model.vit_dim > 0, "ViT dimension must be positive"
        assert config.model.vit_layers > 0, "ViT layers must be positive"
        assert config.model.triplane_resolution > 0, "Triplane resolution must be positive"
        assert config.model.triplane_channels > 0, "Triplane channels must be positive"
        
        # Training validation
        assert config.training.batch_size > 0, "Batch size must be positive"
        assert config.training.num_epochs > 0, "Number of epochs must be positive"
        assert config.training.learning_rate_g > 0, "Generator learning rate must be positive"
        assert config.training.learning_rate_d > 0, "Discriminator learning rate must be positive"
        assert 0 <= config.training.beta1 < 1, "Beta1 must be in [0, 1)"
        assert 0 <= config.training.beta2 < 1, "Beta2 must be in [0, 1)"
        
        # Data validation
        assert config.data.image_size > 0, "Image size must be positive"
        assert config.data.num_workers >= 0, "Number of workers must be non-negative"
        assert 0 <= config.data.val_split <= 1, "Validation split must be in [0, 1]"
        
        # Inference validation
        assert config.inference.batch_size > 0, "Inference batch size must be positive"
        assert config.inference.max_batch_size >= config.inference.batch_size, \
            "Max batch size must be >= batch size"
        
        logger.debug("Configuration validation passed")
    
    def get_device(self, config: AetheristConfig) -> str:
        """Get appropriate device for configuration."""
        device = config.experiment.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def setup_directories(self, config: AetheristConfig) -> None:
        """Create necessary directories for experiment."""
        directories = [
            config.experiment.output_dir,
            config.experiment.checkpoint_dir,
            config.experiment.sample_dir,
            config.experiment.log_dir,
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    
    def print_config(
        self, 
        config: AetheristConfig, 
        sections: Optional[List[str]] = None
    ) -> None:
        """Print configuration in a readable format."""
        if sections is None:
            sections = ["model", "training", "data", "inference", "experiment"]
        
        print("\n" + "="*50)
        print("AETHERIST CONFIGURATION")
        print("="*50)
        
        for section in sections:
            if hasattr(config, section):
                section_config = getattr(config, section)
                print(f"\n[{section.upper()}]")
                print("-" * 30)
                
                for key, value in asdict(section_config).items():
                    if isinstance(value, list) and len(value) > 3:
                        print(f"{key:25}: [{value[0]}, ..., {value[-1]}] ({len(value)} items)")
                    else:
                        print(f"{key:25}: {value}")
        
        print("\n" + "="*50 + "\n")


# Global configuration manager instance
config_manager = ConfigManager()


def load_config(config_path: Optional[str] = None, **kwargs) -> AetheristConfig:
    """Convenience function to load configuration."""
    return config_manager.load_config(config_path, **kwargs)


def save_config(config: AetheristConfig, save_path: str, **kwargs) -> None:
    """Convenience function to save configuration."""
    return config_manager.save_config(config, save_path, **kwargs)