#!/usr/bin/env python3
"""
Configuration-based training script for Aetherist.
Supports YAML configuration files and command-line overrides.
"""

import argparse
import logging
import sys
import torch
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config, ConfigManager
from src.training.training_loop import AetheristTrainer
from src.models.generator import AetheristGenerator
from src.models.discriminator import AetheristDiscriminator


def setup_logging(config):
    """Setup logging based on configuration."""
    log_level = getattr(logging, config.experiment.log_level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # File handler
    log_dir = Path(config.experiment.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "training.log")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def setup_reproducibility(config):
    """Setup reproducibility based on configuration."""
    import random
    import numpy as np
    
    # Set seeds
    torch.manual_seed(config.experiment.seed)
    np.random.seed(config.experiment.seed)
    random.seed(config.experiment.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.experiment.seed)
        torch.cuda.manual_seed_all(config.experiment.seed)
    
    # Set deterministic behavior
    if config.experiment.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True


def create_models(config, device):
    """Create generator and discriminator from configuration."""
    logger = logging.getLogger(__name__)
    
    # Create generator
    logger.info("Creating generator...")
    generator = AetheristGenerator(
        latent_dim=config.model.latent_dim,
        vit_dim=config.model.vit_dim,
        vit_layers=config.model.vit_layers,
        triplane_resolution=config.model.triplane_resolution,
        triplane_channels=config.model.triplane_channels,
    ).to(device)
    
    # Create discriminator
    logger.info("Creating discriminator...")
    discriminator = AetheristDiscriminator(
        input_size=config.data.image_size,
        input_channels=3,
        base_channels=config.model.discriminator_base_channels,
        feature_dim=config.model.discriminator_feature_dim,
        use_multiscale=config.model.use_multiscale_discriminator,
        use_consistency_branch=config.model.use_consistency_branch,
        num_views=config.model.num_views_consistency,
    ).to(device)
    
    # Model compilation
    if config.experiment.compile_model and hasattr(torch, 'compile'):
        logger.info("Compiling models...")
        generator = torch.compile(generator)
        discriminator = torch.compile(discriminator)
    
    # Log model statistics
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    logger.info(f"Generator parameters: {gen_params:,}")
    logger.info(f"Discriminator parameters: {disc_params:,}")
    
    return generator, discriminator


def create_trainer(config, generator, discriminator, device):
    """Create trainer from configuration."""
    logger = logging.getLogger(__name__)
    
    logger.info("Creating trainer...")
    trainer = AetheristTrainer(
        generator=generator,
        discriminator=discriminator,
        device=device,
        # Loss configuration
        adversarial_loss_type="non_saturating",
        lambda_perceptual=config.training.lambda_perceptual,
        lambda_consistency=config.training.lambda_consistency,
        lambda_r1=config.training.lambda_r1,
        # Optimization
        generator_lr=config.training.learning_rate_g,
        discriminator_lr=config.training.learning_rate_d,
        # Multi-view training
        num_views=config.training.num_views,
        view_probability=config.training.view_probability,
        # Logging
        log_every=config.training.log_every,
        save_every=config.training.save_every,
        sample_every=config.training.sample_every,
        # Directories
        checkpoint_dir=config.experiment.checkpoint_dir,
        sample_dir=config.experiment.sample_dir,
        # Experiment tracking
        use_wandb=config.experiment.use_wandb,
        wandb_project=config.experiment.wandb_project,
        wandb_entity=config.experiment.wandb_entity,
    )
    
    return trainer


def create_dataloader(config):
    """Create data loader from configuration."""
    # This would integrate with your dataset implementation
    # For now, return None as placeholder
    logger = logging.getLogger(__name__)
    logger.warning("Dataset creation not implemented - using placeholder")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Train Aetherist with YAML configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration arguments
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/base.yaml",
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--experiment-name", "-n",
        type=str,
        default=None,
        help="Override experiment name"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Override output directory"
    )
    
    # Training overrides
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=None,
        help="Override learning rate (for both G and D)"
    )
    
    # Hardware options
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device (cuda/cpu)"
    )
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Disable mixed precision training"
    )
    
    # Utility options
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Setup everything but don't start training"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint"
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate configuration and exit"
    )
    
    # Command-line overrides
    parser.add_argument(
        "--override", "-O",
        action="append",
        default=[],
        help="Configuration overrides (e.g., -O training.batch_size=32)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(
            config_path=args.config,
            overrides=args.override
        )
        
        # Apply command-line overrides
        if args.experiment_name:
            config.experiment.name = args.experiment_name
        if args.output_dir:
            config.experiment.output_dir = args.output_dir
        if args.batch_size:
            config.training.batch_size = args.batch_size
        if args.epochs:
            config.training.num_epochs = args.epochs
        if args.learning_rate:
            config.training.learning_rate_g = args.learning_rate
            config.training.learning_rate_d = args.learning_rate
        if args.device:
            config.experiment.device = args.device
        if args.no_mixed_precision:
            config.experiment.mixed_precision = False
        
        # Create experiment configuration
        if args.experiment_name or args.output_dir:
            config = config_manager.create_experiment_config(
                config,
                config.experiment.name,
                args.output_dir
            )
        
        # Validate configuration
        if args.validate_config:
            print("✅ Configuration validation successful!")
            config_manager.print_config(config)
            return
        
        # Setup directories
        config_manager.setup_directories(config)
        
        # Setup logging
        setup_logging(config)
        logger = logging.getLogger(__name__)
        
        # Save configuration
        config_path = Path(config.experiment.output_dir) / config.experiment.name / "config.yaml"
        config_manager.save_config(config, config_path)
        logger.info(f"Configuration saved to {config_path}")
        
        # Print configuration
        config_manager.print_config(config)
        
        # Setup reproducibility
        setup_reproducibility(config)
        logger.info("Reproducibility configured")
        
        # Get device
        device = config_manager.get_device(config)
        logger.info(f"Using device: {device}")
        
        # Create models
        generator, discriminator = create_models(config, device)
        
        # Create trainer
        trainer = create_trainer(config, generator, discriminator, device)
        
        # Create data loader
        dataloader = create_dataloader(config)
        
        if args.dry_run:
            logger.info("Dry run completed successfully!")
            return
        
        # Resume from checkpoint if specified
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            trainer.load_checkpoint(args.resume)
        
        # Start training
        logger.info("Starting training...")
        if dataloader is not None:
            trainer.train(dataloader, config.training.num_epochs)
        else:
            logger.error("No dataloader available - cannot start training")
            logger.info("This is expected in the current implementation")
        
        logger.info("Training completed!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()