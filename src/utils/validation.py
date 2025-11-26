#!/usr/bin/env python3
"""Input validation and error handling utilities for Aetherist.

Provides comprehensive validation for all system inputs including:
- Model configurations
- Training parameters
- Inference inputs
- File paths and data formats
"""

import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import logging
from PIL import Image
import json

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

class ConfigurationError(ValidationError):
    """Exception for configuration validation errors."""
    pass

class InputValidationError(ValidationError):
    """Exception for input validation errors."""
    pass

class ModelValidationError(ValidationError):
    """Exception for model validation errors."""
    pass

def validate_tensor_shape(tensor: torch.Tensor, expected_shape: Tuple[int, ...], 
                         name: str = "tensor", allow_batch_dim: bool = True) -> None:
    """Validate tensor has expected shape.
    
    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape (excluding batch dimension if allow_batch_dim=True)
        name: Name of tensor for error messages
        allow_batch_dim: If True, allows additional batch dimension at start
    
    Raises:
        InputValidationError: If shape doesn't match
    """
    if not isinstance(tensor, torch.Tensor):
        raise InputValidationError(f"{name} must be a torch.Tensor, got {type(tensor)}")
        
    actual_shape = tensor.shape
    
    if allow_batch_dim and len(actual_shape) == len(expected_shape) + 1:
        # Check shape excluding batch dimension
        if actual_shape[1:] != expected_shape:
            raise InputValidationError(
                f"{name} has invalid shape {actual_shape}, expected batch dimension + {expected_shape}"
            )
    elif actual_shape != expected_shape:
        raise InputValidationError(
            f"{name} has invalid shape {actual_shape}, expected {expected_shape}"
        )
        
def validate_tensor_range(tensor: torch.Tensor, min_val: float = None, 
                         max_val: float = None, name: str = "tensor") -> None:
    """Validate tensor values are in expected range.
    
    Args:
        tensor: Tensor to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of tensor for error messages
    
    Raises:
        InputValidationError: If values are out of range
    """
    if not isinstance(tensor, torch.Tensor):
        raise InputValidationError(f"{name} must be a torch.Tensor")
        
    if torch.isnan(tensor).any():
        raise InputValidationError(f"{name} contains NaN values")
        
    if torch.isinf(tensor).any():
        raise InputValidationError(f"{name} contains infinite values")
        
    if min_val is not None and tensor.min().item() < min_val:
        raise InputValidationError(
            f"{name} has values below minimum {min_val}: {tensor.min().item()}"
        )
        
    if max_val is not None and tensor.max().item() > max_val:
        raise InputValidationError(
            f"{name} has values above maximum {max_val}: {tensor.max().item()}"
        )

def validate_image_tensor(tensor: torch.Tensor, name: str = "image", 
                         normalize: bool = True) -> None:
    """Validate image tensor format.
    
    Args:
        tensor: Image tensor to validate
        name: Name for error messages
        normalize: If True, expect values in [-1, 1], else [0, 1]
    
    Raises:
        InputValidationError: If tensor is not valid image format
    """
    if not isinstance(tensor, torch.Tensor):
        raise InputValidationError(f"{name} must be a torch.Tensor")
        
    # Check dimensions
    if len(tensor.shape) < 3:
        raise InputValidationError(
            f"{name} must have at least 3 dimensions (C, H, W), got {len(tensor.shape)}"
        )
        
    if len(tensor.shape) == 4:  # Batch dimension
        channels = tensor.shape[1]
    else:
        channels = tensor.shape[0]
        
    if channels not in [1, 3, 4]:
        raise InputValidationError(
            f"{name} must have 1, 3, or 4 channels, got {channels}"
        )
        
    # Check value range
    if normalize:
        validate_tensor_range(tensor, -1.0, 1.0, name)
    else:
        validate_tensor_range(tensor, 0.0, 1.0, name)

def validate_camera_parameters(camera_params: torch.Tensor, name: str = "camera_params") -> None:
    """Validate camera parameter tensor.
    
    Args:
        camera_params: Camera parameters (4x4 matrix or 16-element vector)
        name: Name for error messages
    
    Raises:
        InputValidationError: If camera parameters are invalid
    """
    validate_tensor_shape(camera_params, (16,), name, allow_batch_dim=True)
    validate_tensor_range(camera_params, name=name)  # Check for NaN/inf
    
    # Additional validation for camera matrix properties could be added here
    # e.g., checking if the matrix is a valid transformation matrix

def validate_latent_code(latent_code: torch.Tensor, expected_dim: int, 
                        name: str = "latent_code") -> None:
    """Validate latent code tensor.
    
    Args:
        latent_code: Latent code tensor
        expected_dim: Expected latent dimension
        name: Name for error messages
    
    Raises:
        InputValidationError: If latent code is invalid
    """
    validate_tensor_shape(latent_code, (expected_dim,), name, allow_batch_dim=True)
    validate_tensor_range(latent_code, name=name)

def validate_file_path(path: Union[str, Path], must_exist: bool = True, 
                      file_extensions: List[str] = None, name: str = "file") -> Path:
    """Validate file path.
    
    Args:
        path: Path to validate
        must_exist: If True, file must exist
        file_extensions: List of allowed extensions (e.g., ['.jpg', '.png'])
        name: Name for error messages
    
    Returns:
        Validated Path object
    
    Raises:
        ValidationError: If path is invalid
    """
    if isinstance(path, str):
        path = Path(path)
    elif not isinstance(path, Path):
        raise ValidationError(f"{name} must be a string or Path object")
        
    if must_exist and not path.exists():
        raise ValidationError(f"{name} does not exist: {path}")
        
    if file_extensions is not None:
        if path.suffix.lower() not in [ext.lower() for ext in file_extensions]:
            raise ValidationError(
                f"{name} has invalid extension {path.suffix}, expected one of {file_extensions}"
            )
            
    return path

def validate_directory_path(path: Union[str, Path], must_exist: bool = True, 
                           create_if_missing: bool = False, name: str = "directory") -> Path:
    """Validate directory path.
    
    Args:
        path: Directory path to validate
        must_exist: If True, directory must exist
        create_if_missing: If True, create directory if it doesn't exist
        name: Name for error messages
    
    Returns:
        Validated Path object
    
    Raises:
        ValidationError: If path is invalid
    """
    if isinstance(path, str):
        path = Path(path)
    elif not isinstance(path, Path):
        raise ValidationError(f"{name} must be a string or Path object")
        
    if not path.exists():
        if create_if_missing:
            try:
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path}")
            except Exception as e:
                raise ValidationError(f"Failed to create {name}: {e}")
        elif must_exist:
            raise ValidationError(f"{name} does not exist: {path}")
    elif not path.is_dir():
        raise ValidationError(f"{name} is not a directory: {path}")
        
    return path

def validate_image_file(path: Union[str, Path], name: str = "image") -> Path:
    """Validate image file.
    
    Args:
        path: Image file path
        name: Name for error messages
    
    Returns:
        Validated Path object
    
    Raises:
        ValidationError: If image file is invalid
    """
    path = validate_file_path(
        path, must_exist=True, 
        file_extensions=[".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
        name=name
    )
    
    # Try to load image to verify it's valid
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as e:
        raise ValidationError(f"Invalid image file {name}: {e}")
        
    return path

def validate_config_dict(config: Dict[str, Any], required_keys: List[str],
                        optional_keys: List[str] = None, name: str = "config") -> None:
    """Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        required_keys: List of required keys
        optional_keys: List of optional keys
        name: Name for error messages
    
    Raises:
        ConfigurationError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ConfigurationError(f"{name} must be a dictionary")
        
    # Check required keys
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ConfigurationError(
            f"{name} missing required keys: {missing_keys}"
        )
        
    # Check for unexpected keys
    if optional_keys is not None:
        allowed_keys = set(required_keys + optional_keys)
        unexpected_keys = [key for key in config.keys() if key not in allowed_keys]
        if unexpected_keys:
            logger.warning(
                f"{name} has unexpected keys: {unexpected_keys}"
            )

def validate_positive_number(value: Union[int, float], name: str = "value",
                           integer_only: bool = False) -> Union[int, float]:
    """Validate positive number.
    
    Args:
        value: Value to validate
        name: Name for error messages
        integer_only: If True, require integer
    
    Returns:
        Validated value
    
    Raises:
        ValidationError: If value is invalid
    """
    if integer_only and not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer, got {type(value)}")
    elif not integer_only and not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value)}")
        
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")
        
    return value

def validate_range(value: Union[int, float], min_val: float, max_val: float,
                  name: str = "value") -> Union[int, float]:
    """Validate value is in range.
    
    Args:
        value: Value to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name for error messages
    
    Returns:
        Validated value
    
    Raises:
        ValidationError: If value is out of range
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be a number, got {type(value)}")
        
    if value < min_val or value > max_val:
        raise ValidationError(
            f"{name} must be in range [{min_val}, {max_val}], got {value}"
        )
        
    return value

def validate_model_config(config: Any) -> None:
    """Validate model configuration object.
    
    Args:
        config: Configuration object to validate
    
    Raises:
        ModelValidationError: If configuration is invalid
    """
    required_attrs = ["latent_dim", "triplane_dim", "triplane_res"]
    
    for attr in required_attrs:
        if not hasattr(config, attr):
            raise ModelValidationError(f"Config missing required attribute: {attr}")
            
        value = getattr(config, attr)
        if not isinstance(value, int) or value <= 0:
            raise ModelValidationError(
                f"Config attribute {attr} must be positive integer, got {value}"
            )

def validate_training_config(config: Any) -> None:
    """Validate training configuration.
    
    Args:
        config: Training configuration object
    
    Raises:
        ConfigurationError: If configuration is invalid
    """
    # Required attributes with validation
    validations = {
        "batch_size": lambda x: validate_positive_number(x, "batch_size", integer_only=True),
        "learning_rate": lambda x: validate_range(x, 1e-6, 1.0, "learning_rate"),
        "num_epochs": lambda x: validate_positive_number(x, "num_epochs", integer_only=True),
    }
    
    for attr, validator in validations.items():
        if not hasattr(config, attr):
            raise ConfigurationError(f"Training config missing required attribute: {attr}")
            
        try:
            validator(getattr(config, attr))
        except ValidationError as e:
            raise ConfigurationError(f"Invalid training config: {e}")

def validate_json_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Validate and load JSON configuration file.
    
    Args:
        config_path: Path to JSON configuration file
    
    Returns:
        Loaded configuration dictionary
    
    Raises:
        ConfigurationError: If configuration is invalid
    """
    path = validate_file_path(config_path, must_exist=True, 
                             file_extensions=[".json"], name="config file")
    
    try:
        with open(path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigurationError(f"Invalid JSON in config file {path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to read config file {path}: {e}")
        
    return config

def safe_tensor_operation(operation, *args, **kwargs):
    """Safely execute tensor operation with error handling.
    
    Args:
        operation: Function to execute
        *args: Arguments to pass to operation
        **kwargs: Keyword arguments to pass to operation
    
    Returns:
        Result of operation
    
    Raises:
        ValidationError: If operation fails with validation error
        RuntimeError: For other runtime errors
    """
    try:
        result = operation(*args, **kwargs)
        
        # Check for NaN/inf in result if it's a tensor
        if isinstance(result, torch.Tensor):
            if torch.isnan(result).any():
                raise ValidationError(f"Operation {operation.__name__} produced NaN values")
            if torch.isinf(result).any():
                raise ValidationError(f"Operation {operation.__name__} produced infinite values")
                
        return result
        
    except torch.cuda.OutOfMemoryError:
        raise RuntimeError(
            f"CUDA out of memory during operation {operation.__name__}. "
            "Try reducing batch size or model size."
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise RuntimeError(
                f"Out of memory during operation {operation.__name__}. "
                "Try reducing batch size or model size."
            )
        else:
            raise
    except Exception as e:
        logger.error(f"Operation {operation.__name__} failed: {e}")
        raise

class ValidationContext:
    """Context manager for validation with error aggregation."""
    
    def __init__(self, raise_on_error: bool = True):
        self.errors = []
        self.raise_on_error = raise_on_error
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.errors and self.raise_on_error:
            error_msg = "\n".join([f"- {error}" for error in self.errors])
            raise ValidationError(f"Multiple validation errors:\n{error_msg}")
            
    def validate(self, validation_func, *args, **kwargs):
        """Run validation function and collect errors."""
        try:
            return validation_func(*args, **kwargs)
        except ValidationError as e:
            self.errors.append(str(e))
            if self.raise_on_error:
                raise
        except Exception as e:
            self.errors.append(f"Unexpected error: {e}")
            if self.raise_on_error:
                raise ValidationError(f"Unexpected error: {e}")
                
    @property
    def has_errors(self) -> bool:
        """Check if any errors were collected."""
        return len(self.errors) > 0
        
    def get_errors(self) -> List[str]:
        """Get list of collected errors."""
        return self.errors.copy()
