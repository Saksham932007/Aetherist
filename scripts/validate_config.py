#!/usr/bin/env python3
"""
Configuration validation and testing script for Aetherist.
Validates YAML configurations and tests configuration loading.
"""

import argparse
import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import ConfigManager, load_config
from src.config.config_manager import AetheristConfig


def test_default_config():
    """Test default configuration creation."""
    print("🧪 Testing default configuration...")
    
    try:
        config = AetheristConfig()
        print("✅ Default configuration created successfully")
        return True
    except Exception as e:
        print(f"❌ Default configuration failed: {e}")
        return False


def test_config_loading(config_path: str):
    """Test loading configuration from file."""
    print(f"🧪 Testing configuration loading from {config_path}...")
    
    try:
        config = load_config(config_path)
        print("✅ Configuration loaded successfully")
        return config, True
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def test_config_overrides(config_path: str, overrides: list):
    """Test configuration with overrides."""
    print(f"🧪 Testing configuration overrides...")
    
    try:
        manager = ConfigManager()
        config = manager.load_config(config_path, overrides=overrides)
        print("✅ Configuration with overrides loaded successfully")
        
        # Verify some overrides were applied
        for override in overrides:
            key, value = override.split('=')
            print(f"  Override {key} = {value}")
        
        return config, True
    except Exception as e:
        print(f"❌ Configuration overrides failed: {e}")
        return None, False


def test_config_validation(config):
    """Test configuration validation."""
    print("🧪 Testing configuration validation...")
    
    try:
        manager = ConfigManager()
        manager._validate_config(config)
        print("✅ Configuration validation passed")
        return True
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return False


def test_config_saving(config, output_path: str):
    """Test configuration saving."""
    print(f"🧪 Testing configuration saving to {output_path}...")
    
    try:
        manager = ConfigManager()
        manager.save_config(config, output_path)
        print("✅ Configuration saved successfully")
        
        # Try to load it back
        reloaded_config = manager.load_config(output_path)
        print("✅ Configuration round-trip successful")
        return True
    except Exception as e:
        print(f"❌ Configuration saving failed: {e}")
        return False


def test_experiment_config(config, exp_name: str):
    """Test experiment configuration creation."""
    print(f"🧪 Testing experiment configuration for '{exp_name}'...")
    
    try:
        manager = ConfigManager()
        exp_config = manager.create_experiment_config(
            config, 
            exp_name,
            overrides={"training.batch_size": 16, "model.latent_dim": 128}
        )
        print("✅ Experiment configuration created successfully")
        print(f"  Experiment name: {exp_config.experiment.name}")
        print(f"  Checkpoint dir: {exp_config.experiment.checkpoint_dir}")
        print(f"  Batch size override: {exp_config.training.batch_size}")
        return True
    except Exception as e:
        print(f"❌ Experiment configuration failed: {e}")
        return False


def validate_all_configs():
    """Validate all configuration files in configs directory."""
    print("🧪 Validating all configuration files...")
    
    config_dir = Path("configs")
    if not config_dir.exists():
        print(f"❌ Config directory not found: {config_dir}")
        return False
    
    config_files = list(config_dir.glob("*.yaml"))
    if not config_files:
        print("❌ No YAML configuration files found")
        return False
    
    all_valid = True
    for config_file in config_files:
        print(f"\n📋 Validating {config_file.name}...")
        config, success = test_config_loading(str(config_file))
        if success and config:
            validation_success = test_config_validation(config)
            if validation_success:
                print(f"✅ {config_file.name} is valid")
            else:
                print(f"❌ {config_file.name} validation failed")
                all_valid = False
        else:
            print(f"❌ {config_file.name} loading failed")
            all_valid = False
    
    return all_valid


def print_config_summary(config):
    """Print a summary of the configuration."""
    print("\n📊 Configuration Summary:")
    print("=" * 50)
    
    # Model summary
    print(f"Model:")
    print(f"  Latent dim: {config.model.latent_dim}")
    print(f"  ViT layers: {config.model.vit_layers}")
    print(f"  Triplane: {config.model.triplane_resolution}x{config.model.triplane_resolution}")
    
    # Training summary
    print(f"Training:")
    print(f"  Batch size: {config.training.batch_size}")
    print(f"  Epochs: {config.training.num_epochs}")
    print(f"  Learning rate: G={config.training.learning_rate_g}, D={config.training.learning_rate_d}")
    
    # Data summary
    print(f"Data:")
    print(f"  Image size: {config.data.image_size}")
    print(f"  Dataset: {config.data.dataset_path}")
    
    # Experiment summary
    print(f"Experiment:")
    print(f"  Name: {config.experiment.name}")
    print(f"  Output dir: {config.experiment.output_dir}")
    print(f"  Use wandb: {config.experiment.use_wandb}")
    
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Validate Aetherist configurations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Specific configuration file to test"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Validate all configuration files"
    )
    parser.add_argument(
        "--overrides", "-o",
        nargs="+",
        default=[],
        help="Test overrides (e.g., training.batch_size=16)"
    )
    parser.add_argument(
        "--save-test",
        type=str,
        default=None,
        help="Test saving configuration to specified path"
    )
    parser.add_argument(
        "--experiment-test",
        type=str,
        default="test_experiment",
        help="Test experiment configuration creation"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🔧 Aetherist Configuration Validation")
    print("=" * 50)
    
    success = True
    config = None
    
    # Test default configuration
    if not test_default_config():
        success = False
    
    # Validate all configs or specific config
    if args.all:
        if not validate_all_configs():
            success = False
    elif args.config:
        config, load_success = test_config_loading(args.config)
        if not load_success:
            success = False
        elif config and not test_config_validation(config):
            success = False
    else:
        # Test with base config
        config, load_success = test_config_loading("configs/base.yaml")
        if not load_success:
            success = False
        elif config and not test_config_validation(config):
            success = False
    
    # Test overrides if specified and config is available
    if args.overrides and config:
        override_config, override_success = test_config_overrides(
            args.config or "configs/base.yaml", 
            args.overrides
        )
        if not override_success:
            success = False
        else:
            config = override_config
    
    # Test saving if specified
    if args.save_test and config:
        if not test_config_saving(config, args.save_test):
            success = False
    
    # Test experiment configuration
    if config:
        if not test_experiment_config(config, args.experiment_test):
            success = False
    
    # Print configuration summary
    if config and args.verbose:
        print_config_summary(config)
    
    # Final result
    print("\n" + "=" * 50)
    if success:
        print("🎉 All configuration tests passed!")
        return 0
    else:
        print("❌ Some configuration tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())