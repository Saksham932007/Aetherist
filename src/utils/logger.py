"""
Logging utilities for Aetherist.
Provides unified logging interface with WandB integration and local console output.
"""

import os
import sys
import time
from typing import Any, Dict, Optional, Union
from pathlib import Path
import logging

import torch
import wandb
from omegaconf import DictConfig, OmegaConf


class AetheristLogger:
    """
    Unified logger for Aetherist that handles both local console logging and WandB tracking.
    
    This logger provides a consistent interface for logging metrics, images, and other
    artifacts during training and evaluation. It automatically handles WandB initialization
    and provides fallbacks when WandB is not available.
    """
    
    def __init__(
        self,
        config: DictConfig,
        project_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        log_dir: Optional[str] = None,
        use_wandb: bool = True,
        resume_id: Optional[str] = None,
    ):
        """
        Initialize the logger.
        
        Args:
            config: Full configuration object
            project_name: WandB project name (overrides config)
            experiment_name: Experiment name (overrides config)
            log_dir: Local logging directory (overrides config)
            use_wandb: Whether to use WandB for logging
            resume_id: WandB run ID to resume from
        """
        self.config = config
        self.use_wandb = use_wandb and self._is_wandb_available()
        
        # Set up logging parameters
        self.project_name = project_name or config.logging.project_name
        self.experiment_name = experiment_name or config.logging.experiment_name
        self.log_dir = Path(log_dir) if log_dir else Path(config.paths.log_dir)
        
        # Initialize local logging
        self._setup_local_logging()
        
        # Initialize WandB if available
        self.wandb_run = None
        if self.use_wandb:
            self._setup_wandb(resume_id)
        
        # Tracking variables
        self.step = 0
        self.start_time = time.time()
        
        # Log configuration
        self.info("Logger initialized successfully")
        self.log_config()
    
    def _is_wandb_available(self) -> bool:
        """Check if WandB is available and properly configured."""
        try:
            import wandb
            return True
        except ImportError:
            return False
    
    def _setup_local_logging(self) -> None:
        """Set up local console and file logging."""
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging format
        log_format = "[%(asctime)s][%(levelname)s] %(message)s"
        date_format = "%Y-%m-%d %H:%M:%S"
        
        # Configure root logger
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            datefmt=date_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(self.log_dir / f"{self.experiment_name}.log")
            ]
        )
        
        self.logger = logging.getLogger("aetherist")
    
    def _setup_wandb(self, resume_id: Optional[str] = None) -> None:
        """Initialize WandB logging."""
        try:
            # Prepare WandB config
            wandb_config = OmegaConf.to_container(self.config, resolve=True)
            
            # Initialize WandB run
            self.wandb_run = wandb.init(
                project=self.project_name,
                name=self.experiment_name,
                config=wandb_config,
                entity=self.config.logging.wandb.entity,
                tags=self.config.logging.wandb.tags,
                notes=self.config.logging.wandb.notes,
                resume="allow" if resume_id else None,
                id=resume_id,
                dir=str(self.log_dir),
            )
            
            self.info(f"WandB initialized: {self.wandb_run.url}")
            
        except Exception as e:
            self.warning(f"Failed to initialize WandB: {e}")
            self.use_wandb = False
    
    def log_config(self) -> None:
        """Log the configuration."""
        self.info("Configuration:")
        config_str = OmegaConf.to_yaml(self.config)
        for line in config_str.split('\n'):
            if line.strip():
                self.info(f"  {line}")
    
    def info(self, message: str) -> None:
        """Log an info message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log a warning message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log an error message."""
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log a debug message."""
        self.logger.debug(message)
    
    def log_metrics(
        self,
        metrics: Dict[str, Union[float, int, torch.Tensor]],
        step: Optional[int] = None,
        commit: bool = True,
    ) -> None:
        """
        Log metrics to both console and WandB.
        
        Args:
            metrics: Dictionary of metric name -> value pairs
            step: Training step (uses internal counter if None)
            commit: Whether to commit the metrics to WandB
        """
        if step is not None:
            self.step = step
        
        # Convert tensors to scalars
        processed_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                processed_metrics[key] = value.detach().cpu().item()
            else:
                processed_metrics[key] = value
        
        # Log to console
        metric_str = " | ".join([f"{k}: {v:.6f}" for k, v in processed_metrics.items()])
        self.info(f"Step {self.step:06d} | {metric_str}")
        
        # Log to WandB
        if self.use_wandb and self.wandb_run:
            try:
                self.wandb_run.log(processed_metrics, step=self.step, commit=commit)
            except Exception as e:
                self.warning(f"Failed to log metrics to WandB: {e}")
    
    def log_images(
        self,
        images: Dict[str, torch.Tensor],
        step: Optional[int] = None,
        max_images: int = 8,
    ) -> None:
        """
        Log images to WandB.
        
        Args:
            images: Dictionary of image name -> tensor pairs (B, C, H, W)
            step: Training step (uses internal counter if None)
            max_images: Maximum number of images to log per batch
        """
        if not self.use_wandb or not self.wandb_run:
            return
        
        if step is not None:
            self.step = step
        
        wandb_images = {}
        for name, image_batch in images.items():
            if isinstance(image_batch, torch.Tensor):
                # Convert to numpy and clamp to [0, 1]
                imgs = image_batch.detach().cpu().clamp(0, 1)
                imgs = imgs[:max_images]  # Limit number of images
                
                # Convert to WandB format
                if imgs.dim() == 4:  # (B, C, H, W)
                    imgs = imgs.permute(0, 2, 3, 1)  # (B, H, W, C)
                
                wandb_images[name] = [wandb.Image(img.numpy()) for img in imgs]
        
        try:
            self.wandb_run.log(wandb_images, step=self.step)
            self.info(f"Logged {len(wandb_images)} image sets to WandB")
        except Exception as e:
            self.warning(f"Failed to log images to WandB: {e}")
    
    def log_model_graph(self, model: torch.nn.Module, input_shape: tuple) -> None:
        """
        Log model architecture to WandB.
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape for the model
        """
        if not self.use_wandb or not self.wandb_run:
            return
        
        try:
            # Create dummy input
            dummy_input = torch.randn(input_shape)
            self.wandb_run.watch(model, log="all", log_freq=1000)
            self.info("Model architecture logged to WandB")
        except Exception as e:
            self.warning(f"Failed to log model graph to WandB: {e}")
    
    def save_checkpoint_info(self, checkpoint_path: str, metrics: Dict[str, float]) -> None:
        """
        Log checkpoint save information.
        
        Args:
            checkpoint_path: Path where checkpoint was saved
            metrics: Current metrics at checkpoint time
        """
        self.info(f"Checkpoint saved: {checkpoint_path}")
        
        if self.use_wandb and self.wandb_run:
            try:
                # Log checkpoint as artifact
                artifact = wandb.Artifact(
                    name=f"model-step-{self.step}",
                    type="model",
                    metadata=metrics
                )
                artifact.add_file(checkpoint_path)
                self.wandb_run.log_artifact(artifact)
            except Exception as e:
                self.warning(f"Failed to log checkpoint artifact: {e}")
    
    def finish(self) -> None:
        """Clean up logger and finish WandB run."""
        if self.use_wandb and self.wandb_run:
            try:
                self.wandb_run.finish()
                self.info("WandB run finished")
            except Exception as e:
                self.warning(f"Failed to finish WandB run: {e}")
        
        # Log total training time
        total_time = time.time() - self.start_time
        self.info(f"Total execution time: {total_time:.2f} seconds")
    
    def get_wandb_url(self) -> Optional[str]:
        """Get the WandB run URL if available."""
        if self.use_wandb and self.wandb_run:
            return self.wandb_run.url
        return None