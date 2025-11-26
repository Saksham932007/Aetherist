#!/usr/bin/env python3
"""Security utilities for Aetherist.

Provides input validation, rate limiting, secure file operations, and production security features.
"""

import os
import time
import hashlib
import secrets
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from functools import wraps
from collections import defaultdict, deque
import threading
from contextlib import contextmanager

try:
    import torch
    import numpy as np
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class SecurityError(Exception):
    """Base exception for security-related errors."""
    pass

class InputValidationError(SecurityError):
    """Raised when input validation fails."""
    pass

class RateLimitExceededError(SecurityError):
    """Raised when rate limit is exceeded."""
    pass

class FileSecurityError(SecurityError):
    """Raised when file security validation fails."""
    pass

class InputValidator:
    """Comprehensive input validation for security."""
    
    @staticmethod
    def validate_tensor_input(tensor: Any, 
                            expected_shape: Optional[tuple] = None,
                            min_value: Optional[float] = None,
                            max_value: Optional[float] = None,
                            allowed_dtypes: Optional[List] = None) -> None:
        """Validate tensor inputs for security and correctness."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for tensor validation")
            
        # Check if input is actually a tensor
        if not isinstance(tensor, torch.Tensor):
            raise InputValidationError(f"Expected tensor, got {type(tensor)}")
            
        # Check for NaN and infinity
        if torch.isnan(tensor).any():
            raise InputValidationError("Tensor contains NaN values")
            
        if torch.isinf(tensor).any():
            raise InputValidationError("Tensor contains infinite values")
            
        # Validate shape
        if expected_shape is not None:
            if tensor.shape != expected_shape:
                raise InputValidationError(
                    f"Shape mismatch: expected {expected_shape}, got {tensor.shape}"
                )
                
        # Validate value range
        if min_value is not None and tensor.min().item() < min_value:
            raise InputValidationError(
                f"Tensor contains values below minimum: {tensor.min().item()} < {min_value}"
            )
            
        if max_value is not None and tensor.max().item() > max_value:
            raise InputValidationError(
                f"Tensor contains values above maximum: {tensor.max().item()} > {max_value}"
            )
            
        # Validate dtype
        if allowed_dtypes is not None and tensor.dtype not in allowed_dtypes:
            raise InputValidationError(
                f"Invalid dtype: {tensor.dtype} not in {allowed_dtypes}"
            )
            
        # Check tensor size (prevent memory exhaustion)
        max_elements = 100_000_000  # 100M elements
        if tensor.numel() > max_elements:
            raise InputValidationError(
                f"Tensor too large: {tensor.numel()} elements > {max_elements}"
            )
            
    @staticmethod
    def validate_image_input(image: Any,
                           max_width: int = 2048,
                           max_height: int = 2048,
                           allowed_channels: List[int] = [1, 3, 4]) -> None:
        """Validate image inputs for security."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for image validation")
            
        if not isinstance(image, torch.Tensor):
            raise InputValidationError(f"Expected tensor image, got {type(image)}")
            
        # Check dimensionality
        if image.dim() not in [3, 4]:  # [C, H, W] or [B, C, H, W]
            raise InputValidationError(
                f"Invalid image dimensions: {image.dim()}, expected 3 or 4"
            )
            
        # Extract dimensions
        if image.dim() == 4:
            _, channels, height, width = image.shape
        else:
            channels, height, width = image.shape
            
        # Validate channels
        if channels not in allowed_channels:
            raise InputValidationError(
                f"Invalid number of channels: {channels}, allowed: {allowed_channels}"
            )
            
        # Validate dimensions
        if width > max_width or height > max_height:
            raise InputValidationError(
                f"Image too large: {width}x{height}, max: {max_width}x{max_height}"
            )
            
        # Validate pixel values
        if image.dtype == torch.uint8:
            min_val, max_val = 0, 255
        else:
            min_val, max_val = -1, 1
            
        InputValidator.validate_tensor_input(
            image, min_value=min_val, max_value=max_val
        )
        
    @staticmethod
    def validate_string_input(text: str,
                            max_length: int = 1000,
                            allowed_chars: Optional[str] = None,
                            forbidden_patterns: Optional[List[str]] = None) -> None:
        """Validate string inputs for security."""
        if not isinstance(text, str):
            raise InputValidationError(f"Expected string, got {type(text)}")
            
        # Check length
        if len(text) > max_length:
            raise InputValidationError(
                f"String too long: {len(text)} > {max_length}"
            )
            
        # Check for null bytes (potential injection)
        if '\x00' in text:
            raise InputValidationError("String contains null bytes")
            
        # Check allowed characters
        if allowed_chars is not None:
            for char in text:
                if char not in allowed_chars:
                    raise InputValidationError(
                        f"Invalid character: '{char}' not in allowed set"
                    )
                    
        # Check forbidden patterns
        if forbidden_patterns is not None:
            for pattern in forbidden_patterns:
                if pattern in text:
                    raise InputValidationError(
                        f"Forbidden pattern found: '{pattern}'"
                    )
                    
    @staticmethod
    def validate_file_path(file_path: str,
                         allowed_extensions: Optional[List[str]] = None,
                         base_directory: Optional[str] = None) -> None:
        """Validate file paths for security."""
        path = Path(file_path)
        
        # Check for path traversal
        if '..' in file_path:
            raise FileSecurityError("Path traversal detected")
            
        # Check for absolute paths (if base_directory specified)
        if base_directory is not None:
            base_path = Path(base_directory).resolve()
            try:
                resolved_path = path.resolve()
                if not str(resolved_path).startswith(str(base_path)):
                    raise FileSecurityError(
                        f"Path outside base directory: {resolved_path}"
                    )
            except Exception:
                raise FileSecurityError("Invalid file path")
                
        # Check file extension
        if allowed_extensions is not None:
            if path.suffix.lower() not in allowed_extensions:
                raise FileSecurityError(
                    f"Invalid file extension: {path.suffix}, allowed: {allowed_extensions}"
                )
                
class RateLimiter:
    """Token bucket rate limiter for API protection."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)
        self.lock = threading.Lock()
        
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed under rate limit."""
        current_time = time.time()
        
        with self.lock:
            # Clean old requests
            user_requests = self.requests[identifier]
            while user_requests and current_time - user_requests[0] > self.window_seconds:
                user_requests.popleft()
                
            # Check if under limit
            if len(user_requests) < self.max_requests:
                user_requests.append(current_time)
                return True
            else:
                return False
                
    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        current_time = time.time()
        
        with self.lock:
            user_requests = self.requests[identifier]
            # Clean old requests
            while user_requests and current_time - user_requests[0] > self.window_seconds:
                user_requests.popleft()
                
            return max(0, self.max_requests - len(user_requests))
            
def rate_limit(limiter: RateLimiter, identifier_func: Callable = None):
    """Decorator for rate limiting functions."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get identifier
            if identifier_func:
                identifier = identifier_func(*args, **kwargs)
            else:
                identifier = "default"
                
            if not limiter.is_allowed(identifier):
                raise RateLimitExceededError(
                    f"Rate limit exceeded for {identifier}. "
                    f"Max {limiter.max_requests} requests per {limiter.window_seconds}s"
                )
                
            return func(*args, **kwargs)
        return wrapper
    return decorator

class SecureFileHandler:
    """Secure file operations with validation and sandboxing."""
    
    def __init__(self, base_directory: str, max_file_size: int = 100 * 1024 * 1024):  # 100MB
        self.base_directory = Path(base_directory).resolve()
        self.max_file_size = max_file_size
        
        # Ensure base directory exists
        self.base_directory.mkdir(parents=True, exist_ok=True)
        
    def validate_path(self, file_path: str) -> Path:
        """Validate and resolve file path."""
        # Basic validation
        InputValidator.validate_file_path(
            file_path, base_directory=str(self.base_directory)
        )
        
        # Resolve path
        path = (self.base_directory / file_path).resolve()
        
        # Ensure it's within base directory
        if not str(path).startswith(str(self.base_directory)):
            raise FileSecurityError("Path outside secure directory")
            
        return path
        
    def read_file(self, file_path: str, mode: str = 'r') -> str:
        """Securely read file."""
        path = self.validate_path(file_path)
        
        # Check file exists
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Check file size
        if path.stat().st_size > self.max_file_size:
            raise FileSecurityError(
                f"File too large: {path.stat().st_size} > {self.max_file_size}"
            )
            
        try:
            with open(path, mode) as f:
                return f.read()
        except Exception as e:
            raise FileSecurityError(f"Failed to read file: {e}")
            
    def write_file(self, file_path: str, content: str, mode: str = 'w') -> None:
        """Securely write file."""
        path = self.validate_path(file_path)
        
        # Check content size
        content_size = len(content.encode('utf-8'))
        if content_size > self.max_file_size:
            raise FileSecurityError(
                f"Content too large: {content_size} > {self.max_file_size}"
            )
            
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(path, mode) as f:
                f.write(content)
        except Exception as e:
            raise FileSecurityError(f"Failed to write file: {e}")
            
    def list_files(self, directory: str = "") -> List[str]:
        """Securely list files in directory."""
        dir_path = self.validate_path(directory)
        
        if not dir_path.exists():
            return []
            
        if not dir_path.is_dir():
            raise FileSecurityError("Path is not a directory")
            
        try:
            files = []
            for item in dir_path.iterdir():
                # Get relative path
                rel_path = item.relative_to(self.base_directory)
                files.append(str(rel_path))
            return files
        except Exception as e:
            raise FileSecurityError(f"Failed to list files: {e}")
            
class HashValidator:
    """Hash-based integrity validation."""
    
    @staticmethod
    def compute_hash(data: bytes, algorithm: str = 'sha256') -> str:
        """Compute hash of data."""
        hasher = hashlib.new(algorithm)
        hasher.update(data)
        return hasher.hexdigest()
        
    @staticmethod
    def validate_hash(data: bytes, expected_hash: str, algorithm: str = 'sha256') -> bool:
        """Validate data against expected hash."""
        computed_hash = HashValidator.compute_hash(data, algorithm)
        return secrets.compare_digest(computed_hash, expected_hash)
        
    @staticmethod
    def hash_file(file_path: str, algorithm: str = 'sha256') -> str:
        """Compute hash of file."""
        hasher = hashlib.new(algorithm)
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
        
class SecurityAuditor:
    """Security auditing and monitoring."""
    
    def __init__(self, log_file: str = "security_audit.log"):
        self.log_file = log_file
        self.audit_logger = self._setup_audit_logger()
        
    def _setup_audit_logger(self) -> logging.Logger:
        """Setup audit logging."""
        audit_logger = logging.getLogger("security_audit")
        audit_logger.setLevel(logging.INFO)
        
        # File handler for audit log
        handler = logging.FileHandler(self.log_file)
        handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        audit_logger.addHandler(handler)
        return audit_logger
        
    def log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event."""
        self.audit_logger.info(
            f"Security Event: {event_type} - {details}"
        )
        
    def log_access_attempt(self, resource: str, user: str, success: bool):
        """Log access attempt."""
        status = "SUCCESS" if success else "FAILED"
        self.log_security_event(
            "ACCESS_ATTEMPT",
            {"resource": resource, "user": user, "status": status}
        )
        
    def log_validation_error(self, error_type: str, details: str):
        """Log validation error."""
        self.log_security_event(
            "VALIDATION_ERROR",
            {"error_type": error_type, "details": details}
        )
        
@contextmanager
def security_context(auditor: SecurityAuditor, operation: str, user: str = "system"):
    """Context manager for security auditing."""
    start_time = time.time()
    
    try:
        auditor.log_security_event(
            "OPERATION_START",
            {"operation": operation, "user": user}
        )
        yield
        
        auditor.log_security_event(
            "OPERATION_SUCCESS",
            {
                "operation": operation, 
                "user": user,
                "duration": time.time() - start_time
            }
        )
        
    except Exception as e:
        auditor.log_security_event(
            "OPERATION_FAILURE",
            {
                "operation": operation, 
                "user": user,
                "error": str(e),
                "duration": time.time() - start_time
            }
        )
        raise
        
class ProductionSecurity:
    """Production security configuration and utilities."""
    
    def __init__(self):
        self.security_config = self._load_security_config()
        self.rate_limiter = RateLimiter(
            max_requests=self.security_config.get("max_requests_per_minute", 60),
            window_seconds=60
        )
        self.auditor = SecurityAuditor()
        
    def _load_security_config(self) -> Dict[str, Any]:
        """Load security configuration."""
        # Default security settings
        config = {
            "max_requests_per_minute": 60,
            "max_file_size_mb": 100,
            "allowed_file_extensions": [".jpg", ".png", ".jpeg", ".bmp"],
            "max_image_size": 2048,
            "enable_audit_logging": True,
            "hash_algorithm": "sha256"
        }
        
        # Try to load from environment or config file
        try:
            import json
            config_file = os.environ.get("AETHERIST_SECURITY_CONFIG", "security_config.json")
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config.update(json.load(f))
        except Exception as e:
            logger.warning(f"Could not load security config: {e}")
            
        return config
        
    def validate_request(self, request_data: Dict[str, Any], user_id: str) -> None:
        """Validate incoming request for security."""
        # Rate limiting
        if not self.rate_limiter.is_allowed(user_id):
            self.auditor.log_security_event(
                "RATE_LIMIT_EXCEEDED",
                {"user_id": user_id}
            )
            raise RateLimitExceededError("Rate limit exceeded")
            
        # Input validation
        for key, value in request_data.items():
            if isinstance(value, str):
                InputValidator.validate_string_input(
                    value,
                    max_length=self.security_config.get("max_string_length", 1000)
                )
            elif TORCH_AVAILABLE and isinstance(value, torch.Tensor):
                InputValidator.validate_tensor_input(value)
                
        self.auditor.log_access_attempt("API", user_id, True)
        
    def secure_model_output(self, output: Any) -> Any:
        """Secure model output before returning."""
        if TORCH_AVAILABLE and isinstance(output, torch.Tensor):
            # Clamp values to prevent extreme outputs
            output = torch.clamp(output, -10, 10)
            
            # Check for anomalous outputs
            if torch.isnan(output).any() or torch.isinf(output).any():
                self.auditor.log_security_event(
                    "ANOMALOUS_OUTPUT",
                    {"has_nan": torch.isnan(output).any().item(),
                     "has_inf": torch.isinf(output).any().item()}
                )
                raise SecurityError("Model produced anomalous output")
                
        return output
        
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage."""
        import re
        
        # Remove or replace unsafe characters
        filename = re.sub(r'[^\w\s.-]', '', filename)
        
        # Limit length
        if len(filename) > 255:
            name, ext = os.path.splitext(filename)
            filename = name[:255-len(ext)] + ext
            
        return filename

# Convenience functions for common security operations
def validate_api_input(data: Dict[str, Any], 
                      max_string_length: int = 1000,
                      max_tensor_elements: int = 1000000) -> None:
    """Quick API input validation."""
    for key, value in data.items():
        if isinstance(value, str):
            InputValidator.validate_string_input(value, max_length=max_string_length)
        elif TORCH_AVAILABLE and isinstance(value, torch.Tensor):
            if value.numel() > max_tensor_elements:
                raise InputValidationError(f"Tensor too large: {value.numel()} elements")
            InputValidator.validate_tensor_input(value)
            
def secure_file_upload(file_data: bytes, 
                      filename: str,
                      allowed_extensions: List[str],
                      max_size: int = 10 * 1024 * 1024) -> None:
    """Validate file upload for security."""
    # Check file size
    if len(file_data) > max_size:
        raise FileSecurityError(f"File too large: {len(file_data)} bytes")
        
    # Validate filename
    InputValidator.validate_file_path(filename, allowed_extensions=allowed_extensions)
    
    # Check for magic bytes (basic file type validation)
    if filename.lower().endswith(('.jpg', '.jpeg')):
        if not file_data.startswith(b'\xff\xd8\xff'):
            raise FileSecurityError("Invalid JPEG file format")
    elif filename.lower().endswith('.png'):
        if not file_data.startswith(b'\x89PNG\r\n\x1a\n'):
            raise FileSecurityError("Invalid PNG file format")
