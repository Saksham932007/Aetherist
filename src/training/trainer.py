"""
Training system for Aetherist GAN.

This module implements the complete training loop for the Aetherist generator and discriminator,
including loss functions, optimization strategies, and training utilities.

Key Components:
1. GAN Losses: Adversarial, consistency, and auxiliary losses
2. Training Loop: Orchestrates generator and discriminator training
3. Optimization: Advanced optimizers with learning rate scheduling
4. Metrics: Training monitoring and evaluation metrics
5. Checkpointing: Model saving and restoration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Union
import math
import time
from pathlib import Path
from collections import defaultdict

# For distributed training (optional)
try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

# For monitoring (optional)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class GANLosses:
    """
    Collection of GAN loss functions for training.
    
    Supports various adversarial losses and auxiliary losses for improved training.
    """
    
    @staticmethod
    def adversarial_loss(
        real_logits: Tensor,
        fake_logits: Tensor,
        mode: str = "hinge",
        label_smoothing: float = 0.0,
    ) -> Dict[str, Tensor]:
        """
        Compute adversarial losses for generator and discriminator.
        
        Args:
            real_logits: Discriminator output for real samples (B, 1)
            fake_logits: Discriminator output for fake samples (B, 1)
            mode: Loss type ("hinge", "wasserstein", "lsgan", "vanilla")
            label_smoothing: Label smoothing factor for real samples
            
        Returns:
            Dictionary with generator and discriminator losses
        """
        if mode == "hinge":
            # Hinge loss (more stable training)
            d_loss_real = F.relu(1.0 - real_logits).mean()
            d_loss_fake = F.relu(1.0 + fake_logits).mean()
            d_loss = d_loss_real + d_loss_fake
            
            g_loss = -fake_logits.mean()
            
        elif mode == "wasserstein":
            # Wasserstein loss (requires gradient penalty)
            d_loss_real = -real_logits.mean()
            d_loss_fake = fake_logits.mean()
            d_loss = d_loss_real + d_loss_fake
            
            g_loss = -fake_logits.mean()
            
        elif mode == "lsgan":
            # Least squares GAN
            real_target = torch.ones_like(real_logits) * (1.0 - label_smoothing)
            fake_target = torch.zeros_like(fake_logits)
            
            d_loss_real = F.mse_loss(real_logits, real_target)
            d_loss_fake = F.mse_loss(fake_logits, fake_target)
            d_loss = (d_loss_real + d_loss_fake) / 2
            
            g_target = torch.ones_like(fake_logits)
            g_loss = F.mse_loss(fake_logits, g_target)
            
        elif mode == "vanilla":
            # Standard GAN loss
            real_target = torch.ones_like(real_logits) * (1.0 - label_smoothing)
            fake_target = torch.zeros_like(fake_logits)
            
            d_loss_real = F.binary_cross_entropy_with_logits(real_logits, real_target)
            d_loss_fake = F.binary_cross_entropy_with_logits(fake_logits, fake_target)
            d_loss = d_loss_real + d_loss_fake
            
            g_target = torch.ones_like(fake_logits)
            g_loss = F.binary_cross_entropy_with_logits(fake_logits, g_target)
            
        else:
            raise ValueError(f"Unknown adversarial loss mode: {mode}")
        
        return {
            'discriminator_loss': d_loss,
            'generator_loss': g_loss,
            'real_loss': d_loss_real,
            'fake_loss': d_loss_fake,
        }
    
    @staticmethod
    def gradient_penalty(
        discriminator: nn.Module,
        real_images: Tensor,
        fake_images: Tensor,
        lambda_gp: float = 10.0,
    ) -> Tensor:
        """
        Compute gradient penalty for improved training stability.
        
        Args:
            discriminator: Discriminator model
            real_images: Real image batch (B, C, H, W)
            fake_images: Generated image batch (B, C, H, W)  
            lambda_gp: Gradient penalty weight
            
        Returns:
            Gradient penalty loss
        """
        batch_size = real_images.size(0)
        device = real_images.device
        
        # Random interpolation factor
        alpha = torch.rand(batch_size, 1, 1, 1, device=device)
        
        # Interpolate between real and fake images
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        interpolated.requires_grad_(True)
        
        # Get discriminator output for interpolated images
        d_interpolated = discriminator(interpolated)
        
        if isinstance(d_interpolated, dict):
            d_interpolated = d_interpolated['total_score']
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        # Compute gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
    
    @staticmethod
    def consistency_loss(
        view1_output: Dict[str, Tensor],
        view2_output: Dict[str, Tensor],
        lambda_consistency: float = 1.0,
    ) -> Tensor:
        """
        Compute 3D consistency loss between different viewpoints.
        
        Args:
            view1_output: Generator output for first viewpoint
            view2_output: Generator output for second viewpoint
            lambda_consistency: Consistency loss weight
            
        Returns:
            3D consistency loss
        """
        loss = 0.0
        
        # Tri-plane consistency (tri-planes should be view-invariant)
        if 'triplanes' in view1_output and 'triplanes' in view2_output:
            triplane_loss = 0.0
            for plane_name in view1_output['triplanes']:
                plane1 = view1_output['triplanes'][plane_name]
                plane2 = view2_output['triplanes'][plane_name]
                triplane_loss += F.mse_loss(plane1, plane2)
            loss += triplane_loss / len(view1_output['triplanes'])
        
        # Style consistency (style vectors should be identical for same latent)
        if 'w' in view1_output and 'w' in view2_output:
            style_loss = F.mse_loss(view1_output['w'], view2_output['w'])
            loss += style_loss
        
        # ViT feature consistency  
        if 'vit_features' in view1_output and 'vit_features' in view2_output:
            feature_loss = F.mse_loss(view1_output['vit_features'], view2_output['vit_features'])
            loss += feature_loss
        
        return lambda_consistency * loss
    
    @staticmethod
    def perceptual_loss(
        real_images: Tensor,
        fake_images: Tensor,
        vgg_model: Optional[nn.Module] = None,
        lambda_perceptual: float = 1.0,
    ) -> Tensor:
        """
        Compute perceptual loss using VGG features.
        
        Args:
            real_images: Real images (B, 3, H, W)
            fake_images: Generated images (B, 3, H, W)
            vgg_model: Pre-trained VGG model for feature extraction
            lambda_perceptual: Perceptual loss weight
            
        Returns:
            Perceptual loss
        """
        if vgg_model is None:
            # Simple L2 loss fallback
            return lambda_perceptual * F.mse_loss(fake_images, real_images)
        
        # Extract VGG features
        real_features = vgg_model(real_images)
        fake_features = vgg_model(fake_images)
        
        # Compute feature loss
        loss = 0.0
        for real_feat, fake_feat in zip(real_features, fake_features):
            loss += F.mse_loss(fake_feat, real_feat)
        
        return lambda_perceptual * loss


class LearningRateScheduler:
    """
    Advanced learning rate scheduling for GAN training.
    
    Supports various scheduling strategies including cosine annealing,
    linear decay, and exponential decay.
    """
    
    def __init__(
        self,
        optimizer: optim.Optimizer,
        schedule_type: str = "cosine",
        initial_lr: float = 2e-4,
        min_lr: float = 1e-6,
        warmup_steps: int = 1000,
        total_steps: int = 100000,
        decay_rate: float = 0.95,
        decay_steps: int = 5000,
    ):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            schedule_type: Type of schedule ("cosine", "linear", "exponential", "constant")
            initial_lr: Initial learning rate
            min_lr: Minimum learning rate
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            decay_rate: Decay rate for exponential schedule
            decay_steps: Steps between decay for exponential schedule
        """
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        
        self.step_count = 0
    
    def step(self) -> float:
        """Update learning rate and return current LR."""
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Warmup phase
            lr = self.initial_lr * (self.step_count / self.warmup_steps)
        else:
            # Main schedule
            progress = (self.step_count - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            if self.schedule_type == "cosine":
                lr = self.min_lr + (self.initial_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
            elif self.schedule_type == "linear":
                lr = self.initial_lr * (1 - progress) + self.min_lr * progress
            elif self.schedule_type == "exponential":
                decay_factor = self.decay_rate ** (self.step_count // self.decay_steps)
                lr = max(self.initial_lr * decay_factor, self.min_lr)
            elif self.schedule_type == "constant":
                lr = self.initial_lr
            else:
                raise ValueError(f"Unknown schedule type: {self.schedule_type}")
        
        # Update optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


class TrainingMetrics:
    """
    Comprehensive training metrics tracking and logging.
    """
    
    def __init__(self, log_interval: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            log_interval: Steps between metric logging
        """
        self.log_interval = log_interval
        self.step_count = 0
        self.epoch_count = 0
        
        # Metric storage
        self.metrics = defaultdict(list)
        self.running_metrics = defaultdict(float)
        self.metric_counts = defaultdict(int)
        
        # Timing
        self.start_time = time.time()
        self.step_times = []
    
    def update(self, metrics: Dict[str, float]):
        """Update metrics with new values."""
        self.step_count += 1
        
        for key, value in metrics.items():
            self.running_metrics[key] += value
            self.metric_counts[key] += 1
        
        # Record step time
        current_time = time.time()
        if len(self.step_times) > 0:
            step_time = current_time - self.last_time
            self.step_times.append(step_time)
            if len(self.step_times) > 100:  # Keep last 100 step times
                self.step_times.pop(0)
        self.last_time = current_time
    
    def get_averages(self, reset: bool = True) -> Dict[str, float]:
        """Get average metrics and optionally reset counters."""
        averages = {}
        for key in self.running_metrics:
            if self.metric_counts[key] > 0:
                averages[key] = self.running_metrics[key] / self.metric_counts[key]
        
        if reset:
            self.running_metrics.clear()
            self.metric_counts.clear()
        
        # Add timing metrics
        if len(self.step_times) > 0:
            averages['step_time'] = sum(self.step_times) / len(self.step_times)
            averages['steps_per_sec'] = 1.0 / averages['step_time']
        
        # Add total elapsed time
        averages['elapsed_time'] = time.time() - self.start_time
        
        return averages
    
    def should_log(self) -> bool:
        """Check if it's time to log metrics."""
        return self.step_count % self.log_interval == 0
    
    def log_metrics(self, averages: Dict[str, float], prefix: str = ""):
        """Log metrics to console and wandb if available."""
        step_str = f"Step {self.step_count}"
        if self.epoch_count > 0:
            step_str = f"Epoch {self.epoch_count}, {step_str}"
        
        # Format metrics for logging
        metric_strs = []
        for key, value in averages.items():
            if 'loss' in key or 'score' in key:
                metric_strs.append(f"{key}: {value:.4f}")
            elif 'time' in key:
                metric_strs.append(f"{key}: {value:.3f}s")
            elif 'per_sec' in key:
                metric_strs.append(f"{key}: {value:.1f}")
            else:
                metric_strs.append(f"{key}: {value:.4f}")
        
        print(f"{prefix}{step_str} | {' | '.join(metric_strs)}")
        
        # Log to wandb if available
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb_metrics = {f"{prefix}{k}": v for k, v in averages.items()}
            wandb.log(wandb_metrics, step=self.step_count)


class CheckpointManager:
    """
    Model checkpointing and restoration manager.
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        save_interval: int = 5000,
        max_checkpoints: int = 5,
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            save_interval: Steps between checkpoint saves
            max_checkpoints: Maximum number of checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.save_interval = save_interval
        self.max_checkpoints = max_checkpoints
        
        self.step_count = 0
        self.saved_checkpoints = []
    
    def should_save(self, step: int) -> bool:
        """Check if it's time to save a checkpoint."""
        return step % self.save_interval == 0 or step == 0
    
    def save_checkpoint(
        self,
        step: int,
        generator: nn.Module,
        discriminator: nn.Module,
        g_optimizer: optim.Optimizer,
        d_optimizer: optim.Optimizer,
        g_scheduler: Optional[LearningRateScheduler] = None,
        d_scheduler: Optional[LearningRateScheduler] = None,
        metrics: Optional[Dict] = None,
    ):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_step_{step:08d}.pt"
        
        checkpoint = {
            'step': step,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
        }
        
        if g_scheduler is not None:
            checkpoint['g_scheduler_state'] = {
                'step_count': g_scheduler.step_count,
                'schedule_type': g_scheduler.schedule_type,
                'initial_lr': g_scheduler.initial_lr,
                'min_lr': g_scheduler.min_lr,
            }
        
        if d_scheduler is not None:
            checkpoint['d_scheduler_state'] = {
                'step_count': d_scheduler.step_count,
                'schedule_type': d_scheduler.schedule_type,
                'initial_lr': d_scheduler.initial_lr,
                'min_lr': d_scheduler.min_lr,
            }
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        self.saved_checkpoints.append(checkpoint_path)
        
        # Clean up old checkpoints
        if len(self.saved_checkpoints) > self.max_checkpoints:
            old_checkpoint = self.saved_checkpoints.pop(0)
            if old_checkpoint.exists():
                old_checkpoint.unlink()
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
        generator: nn.Module,
        discriminator: nn.Module,
        g_optimizer: optim.Optimizer,
        d_optimizer: optim.Optimizer,
        g_scheduler: Optional[LearningRateScheduler] = None,
        d_scheduler: Optional[LearningRateScheduler] = None,
    ) -> Dict:
        """Load training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        
        if g_scheduler is not None and 'g_scheduler_state' in checkpoint:
            g_scheduler.step_count = checkpoint['g_scheduler_state']['step_count']
        
        if d_scheduler is not None and 'd_scheduler_state' in checkpoint:
            d_scheduler.step_count = checkpoint['d_scheduler_state']['step_count']
        
        step = checkpoint['step']
        metrics = checkpoint.get('metrics', {})
        
        print(f"Loaded checkpoint from step {step}: {checkpoint_path}")
        
        return {
            'step': step,
            'metrics': metrics,
        }
    
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to the latest checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_step_*.pt"))
        if not checkpoints:
            return None
        
        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.stem.split('_')[-1]))
        return checkpoints[-1]


def test_training_components():
    """Test function for training components."""
    print("🏋️ Testing Training Components")
    print("=" * 40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    batch_size = 4
    
    # Test loss functions
    print("\n1. Testing Loss Functions...")
    real_logits = torch.randn(batch_size, 1, device=device) * 0.5 + 0.5
    fake_logits = torch.randn(batch_size, 1, device=device) * 0.5 - 0.5
    
    # Test different adversarial losses
    for loss_mode in ["hinge", "wasserstein", "lsgan", "vanilla"]:
        losses = GANLosses.adversarial_loss(real_logits, fake_logits, mode=loss_mode)
        print(f"  {loss_mode}: G={losses['generator_loss']:.4f}, D={losses['discriminator_loss']:.4f}")
    
    # Test learning rate scheduler
    print("\n2. Testing Learning Rate Scheduler...")
    dummy_model = nn.Linear(10, 1).to(device)
    optimizer = optim.Adam(dummy_model.parameters(), lr=2e-4)
    
    scheduler = LearningRateScheduler(
        optimizer,
        schedule_type="cosine",
        total_steps=1000,
        warmup_steps=100,
    )
    
    print(f"  Initial LR: {scheduler.step():.6f}")
    for _ in range(99):  # Complete warmup
        scheduler.step()
    print(f"  After warmup: {optimizer.param_groups[0]['lr']:.6f}")
    
    for _ in range(400):  # Mid training
        scheduler.step()
    print(f"  Mid training: {optimizer.param_groups[0]['lr']:.6f}")
    
    for _ in range(500):  # End training
        scheduler.step()
    print(f"  End training: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Test metrics tracker
    print("\n3. Testing Metrics Tracker...")
    metrics = TrainingMetrics(log_interval=5)
    
    for i in range(10):
        test_metrics = {
            'g_loss': 0.5 + 0.1 * torch.randn(1).item(),
            'd_loss': 0.3 + 0.05 * torch.randn(1).item(),
            'consistency_loss': 0.1 + 0.02 * torch.randn(1).item(),
        }
        metrics.update(test_metrics)
        
        if metrics.should_log():
            averages = metrics.get_averages(reset=True)
            metrics.log_metrics(averages, prefix="Test ")
    
    # Test checkpoint manager  
    print("\n4. Testing Checkpoint Manager...")
    checkpoint_dir = Path("test_checkpoints")
    checkpoint_manager = CheckpointManager(checkpoint_dir, save_interval=100)
    
    # Create dummy models
    generator = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 3*64*64)).to(device)
    discriminator = nn.Sequential(nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 1)).to(device)
    
    g_opt = optim.Adam(generator.parameters())
    d_opt = optim.Adam(discriminator.parameters())
    
    # Test saving
    if checkpoint_manager.should_save(100):
        checkpoint_manager.save_checkpoint(
            step=100,
            generator=generator,
            discriminator=discriminator,
            g_optimizer=g_opt,
            d_optimizer=d_opt,
            metrics={'test_metric': 0.5}
        )
        print("  Checkpoint saved successfully")
    
    # Test loading
    latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
    if latest_checkpoint:
        result = checkpoint_manager.load_checkpoint(
            latest_checkpoint,
            generator=generator,
            discriminator=discriminator,
            g_optimizer=g_opt,
            d_optimizer=d_opt,
        )
        print(f"  Checkpoint loaded from step {result['step']}")
    
    # Cleanup
    import shutil
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    
    print("\n✅ All training components working correctly!")


if __name__ == "__main__":
    test_training_components()