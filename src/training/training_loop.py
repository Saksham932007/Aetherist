"""
Main training loop for Aetherist GAN.

This module implements the complete training orchestration, bringing together
the generator, discriminator, losses, and optimization strategies.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Optional, Any
from pathlib import Path
import time

from ..models.generator import AetheristGenerator  
from ..models.discriminator import AetheristDiscriminator
from ..utils.camera import sample_camera_poses, perspective_projection_matrix
from .trainer import GANLosses, LearningRateScheduler, TrainingMetrics, CheckpointManager


class AetheristTrainer:
    """
    Main trainer class for Aetherist GAN.
    
    Orchestrates training between generator and discriminator with advanced
    optimization strategies and comprehensive monitoring.
    """
    
    def __init__(
        self,
        # Model configuration
        generator_config: Dict[str, Any],
        discriminator_config: Dict[str, Any],
        
        # Training configuration
        batch_size: int = 32,
        learning_rate_g: float = 1e-4,
        learning_rate_d: float = 4e-4,
        beta1: float = 0.0,
        beta2: float = 0.99,
        
        # Loss configuration
        adversarial_mode: str = "hinge",
        lambda_consistency: float = 10.0,
        lambda_perceptual: float = 1.0,
        lambda_gradient_penalty: float = 10.0,
        use_gradient_penalty: bool = True,
        
        # Training strategy
        g_steps_per_d_step: int = 1,
        d_steps_per_g_step: int = 1,
        warmup_steps: int = 1000,
        
        # Monitoring
        log_interval: int = 100,
        save_interval: int = 5000,
        sample_interval: int = 1000,
        
        # Paths
        checkpoint_dir: str = "checkpoints",
        sample_dir: str = "samples",
        
        # Device
        device: str = "auto",
    ):
        """
        Initialize Aetherist trainer.
        
        Args:
            generator_config: Configuration for generator model
            discriminator_config: Configuration for discriminator model
            batch_size: Training batch size
            learning_rate_g: Generator learning rate
            learning_rate_d: Discriminator learning rate
            beta1: Adam optimizer beta1 parameter
            beta2: Adam optimizer beta2 parameter
            adversarial_mode: Type of adversarial loss
            lambda_consistency: 3D consistency loss weight
            lambda_perceptual: Perceptual loss weight
            lambda_gradient_penalty: Gradient penalty weight
            use_gradient_penalty: Whether to use gradient penalty
            g_steps_per_d_step: Generator steps per discriminator step
            d_steps_per_g_step: Discriminator steps per generator step
            warmup_steps: Number of warmup steps
            log_interval: Steps between logging
            save_interval: Steps between checkpoint saves
            sample_interval: Steps between sample generation
            checkpoint_dir: Directory for checkpoints
            sample_dir: Directory for sample outputs
            device: Training device
        """
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Training on device: {self.device}")
        
        # Store configuration
        self.config = {
            'generator': generator_config,
            'discriminator': discriminator_config,
            'batch_size': batch_size,
            'learning_rates': {'generator': learning_rate_g, 'discriminator': learning_rate_d},
            'loss_weights': {
                'consistency': lambda_consistency,
                'perceptual': lambda_perceptual,
                'gradient_penalty': lambda_gradient_penalty,
            },
            'training_strategy': {
                'g_steps_per_d': g_steps_per_d_step,
                'd_steps_per_g': d_steps_per_g_step,
                'warmup_steps': warmup_steps,
            }
        }
        
        # Initialize models
        print("Initializing models...")
        self.generator = AetheristGenerator(**generator_config).to(self.device)
        self.discriminator = AetheristDiscriminator(**discriminator_config).to(self.device)
        
        # Print model info
        g_params = sum(p.numel() for p in self.generator.parameters())
        d_params = sum(p.numel() for p in self.discriminator.parameters())
        print(f"Generator parameters: {g_params:,}")
        print(f"Discriminator parameters: {d_params:,}")
        print(f"Total parameters: {g_params + d_params:,}")
        
        # Initialize optimizers
        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=learning_rate_g,
            betas=(beta1, beta2),
        )
        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=learning_rate_d,
            betas=(beta1, beta2),
        )
        
        # Initialize schedulers
        self.g_scheduler = LearningRateScheduler(
            self.g_optimizer,
            schedule_type="cosine",
            initial_lr=learning_rate_g,
            warmup_steps=warmup_steps,
        )
        self.d_scheduler = LearningRateScheduler(
            self.d_optimizer,
            schedule_type="cosine", 
            initial_lr=learning_rate_d,
            warmup_steps=warmup_steps,
        )
        
        # Loss configuration
        self.adversarial_mode = adversarial_mode
        self.lambda_consistency = lambda_consistency
        self.lambda_perceptual = lambda_perceptual
        self.use_gradient_penalty = use_gradient_penalty
        self.lambda_gradient_penalty = lambda_gradient_penalty
        
        # Training strategy
        self.g_steps_per_d_step = g_steps_per_d_step
        self.d_steps_per_g_step = d_steps_per_g_step
        
        # Initialize training utilities
        self.metrics = TrainingMetrics(log_interval=log_interval)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            save_interval=save_interval,
        )
        
        # Create sample directory
        self.sample_dir = Path(sample_dir)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.sample_interval = sample_interval
        
        # Training state
        self.current_step = 0
        self.current_epoch = 0
        
        # Fixed latents for consistent sampling
        self.fixed_z = torch.randn(8, generator_config.get('latent_dim', 512), device=self.device)
        
        print("Trainer initialized successfully!")
    
    def train_step(self, real_images: torch.Tensor) -> Dict[str, float]:
        """
        Execute single training step.
        
        Args:
            real_images: Batch of real images (B, C, H, W)
            
        Returns:
            Dictionary of training metrics
        """
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)
        
        metrics = {}
        
        # Generate random latents
        z = torch.randn(batch_size, self.generator.latent_dim, device=self.device)
        
        # Sample camera poses for 3D consistency
        eye_positions, view_matrices, camera_angles = sample_camera_poses(
            batch_size, device=self.device
        )
        proj_matrices = perspective_projection_matrix(
            fov_degrees=torch.tensor([50.0] * batch_size),
            aspect_ratio=torch.tensor([1.0] * batch_size),
        ).to(self.device)
        camera_matrices = torch.bmm(proj_matrices, view_matrices)
        
        # ================== Discriminator Training ==================
        for d_step in range(self.d_steps_per_g_step):
            self.d_optimizer.zero_grad()
            
            # Generate fake images
            with torch.no_grad():
                fake_output = self.generator(z, camera_matrices)
                fake_images = fake_output['high_res_image']
            
            # Discriminator forward pass
            real_d_output = self.discriminator(real_images)
            fake_d_output = self.discriminator(fake_images.detach())
            
            # Adversarial loss
            adv_losses = GANLosses.adversarial_loss(
                real_d_output['total_score'],
                fake_d_output['total_score'],
                mode=self.adversarial_mode,
            )
            d_loss = adv_losses['discriminator_loss']
            
            # Gradient penalty (if using Wasserstein)
            if self.use_gradient_penalty:
                gp_loss = GANLosses.gradient_penalty(
                    self.discriminator,
                    real_images,
                    fake_images.detach(),
                    lambda_gp=self.lambda_gradient_penalty,
                )
                d_loss += gp_loss
                metrics['gradient_penalty'] = gp_loss.item()
            
            # Backward pass
            d_loss.backward()
            self.d_optimizer.step()
            
            # Record metrics
            metrics.update({
                'd_loss_total': d_loss.item(),
                'd_loss_real': adv_losses['real_loss'].item(),
                'd_loss_fake': adv_losses['fake_loss'].item(),
                'd_score_real': real_d_output['total_score'].mean().item(),
                'd_score_fake': fake_d_output['total_score'].mean().item(),
            })
        
        # ================== Generator Training ==================
        for g_step in range(self.g_steps_per_d_step):
            self.g_optimizer.zero_grad()
            
            # Generate fake images
            fake_output = self.generator(z, camera_matrices, return_triplanes=True)
            fake_images = fake_output['high_res_image']
            
            # Discriminator evaluation
            fake_d_output = self.discriminator(fake_images)
            
            # Adversarial loss
            adv_losses = GANLosses.adversarial_loss(
                torch.zeros_like(fake_d_output['total_score']),  # Dummy real scores
                fake_d_output['total_score'],
                mode=self.adversarial_mode,
            )
            g_loss = adv_losses['generator_loss']
            
            # 3D consistency loss (multi-view)
            if self.lambda_consistency > 0:
                # Generate second view
                z2 = z  # Same latent for consistency
                eye_positions2, view_matrices2, camera_angles2 = sample_camera_poses(
                    batch_size, device=self.device
                )
                proj_matrices2 = perspective_projection_matrix(
                    fov_degrees=torch.tensor([50.0] * batch_size),
                    aspect_ratio=torch.tensor([1.0] * batch_size),
                ).to(self.device)
                camera_matrices2 = torch.bmm(proj_matrices2, view_matrices2)
                
                fake_output2 = self.generator(z2, camera_matrices2, return_triplanes=True)
                
                consistency_loss = GANLosses.consistency_loss(
                    fake_output, fake_output2, self.lambda_consistency
                )
                g_loss += consistency_loss
                metrics['consistency_loss'] = consistency_loss.item()
            
            # Perceptual loss (if available)
            if self.lambda_perceptual > 0 and real_images.size() == fake_images.size():
                perceptual_loss = GANLosses.perceptual_loss(
                    real_images, fake_images, lambda_perceptual=self.lambda_perceptual
                )
                g_loss += perceptual_loss
                metrics['perceptual_loss'] = perceptual_loss.item()
            
            # Backward pass
            g_loss.backward()
            self.g_optimizer.step()
            
            # Record metrics
            metrics.update({
                'g_loss_total': g_loss.item(),
                'g_loss_adv': adv_losses['generator_loss'].item(),
            })
        
        # Update learning rates
        g_lr = self.g_scheduler.step()
        d_lr = self.d_scheduler.step()
        metrics.update({
            'lr_generator': g_lr,
            'lr_discriminator': d_lr,
        })
        
        return metrics
    
    def train(
        self,
        dataloader: DataLoader,
        num_epochs: int,
        resume_from_checkpoint: Optional[str] = None,
    ):
        """
        Main training loop.
        
        Args:
            dataloader: Training data loader
            num_epochs: Number of training epochs
            resume_from_checkpoint: Path to checkpoint to resume from
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Batches per epoch: {len(dataloader)}")
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            checkpoint_data = self.checkpoint_manager.load_checkpoint(
                resume_from_checkpoint,
                self.generator,
                self.discriminator,
                self.g_optimizer,
                self.d_optimizer,
                self.g_scheduler,
                self.d_scheduler,
            )
            self.current_step = checkpoint_data['step']
            print(f"Resumed training from step {self.current_step}")
        
        # Training loop
        self.generator.train()
        self.discriminator.train()
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            for batch_idx, batch in enumerate(dataloader):
                # Extract images from batch (handle different dataset formats)
                if isinstance(batch, dict):
                    real_images = batch['images']
                elif isinstance(batch, (list, tuple)):
                    real_images = batch[0]
                else:
                    real_images = batch
                
                # Training step
                step_metrics = self.train_step(real_images)
                self.metrics.update(step_metrics)
                self.current_step += 1
                
                # Logging
                if self.metrics.should_log():
                    averages = self.metrics.get_averages(reset=True)
                    self.metrics.log_metrics(averages, prefix=f"E{epoch:03d} ")
                
                # Sample generation
                if self.current_step % self.sample_interval == 0:
                    self.generate_samples()
                
                # Checkpointing
                if self.checkpoint_manager.should_save(self.current_step):
                    self.checkpoint_manager.save_checkpoint(
                        step=self.current_step,
                        generator=self.generator,
                        discriminator=self.discriminator,
                        g_optimizer=self.g_optimizer,
                        d_optimizer=self.d_optimizer,
                        g_scheduler=self.g_scheduler,
                        d_scheduler=self.d_scheduler,
                        metrics={'epoch': epoch, 'batch': batch_idx},
                    )
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s")
        
        print("Training completed!")
    
    def generate_samples(self):
        """Generate and save sample images."""
        self.generator.eval()
        
        with torch.no_grad():
            # Sample with fixed latents for consistency
            batch_size = self.fixed_z.size(0)
            
            # Random camera poses
            eye_positions, view_matrices, camera_angles = sample_camera_poses(
                batch_size, device=self.device
            )
            proj_matrices = perspective_projection_matrix(
                fov_degrees=torch.tensor([50.0] * batch_size),
                aspect_ratio=torch.tensor([1.0] * batch_size),
            ).to(self.device)
            camera_matrices = torch.bmm(proj_matrices, view_matrices)
            
            # Generate samples
            output = self.generator(self.fixed_z, camera_matrices, return_triplanes=True)
            samples = output['high_res_image']
            
            # Save samples
            sample_path = self.sample_dir / f"samples_step_{self.current_step:08d}.png"
            self._save_image_grid(samples, sample_path)
        
        self.generator.train()
    
    def _save_image_grid(self, images: torch.Tensor, path: Path):
        """Save a grid of images."""
        # Convert from [-1, 1] to [0, 1] 
        images = torch.clamp((images + 1) / 2, 0, 1)
        
        # Simple grid saving (can be enhanced with torchvision.utils.save_image)
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')
        
        batch_size = images.size(0)
        grid_size = int(batch_size ** 0.5)
        if grid_size * grid_size < batch_size:
            grid_size += 1
        
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
        axes = axes.flatten() if batch_size > 1 else [axes]
        
        for i in range(batch_size):
            img = images[i].cpu().permute(1, 2, 0).numpy()
            axes[i].imshow(img)
            axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(batch_size, grid_size * grid_size):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved samples: {path}")


def create_trainer_from_config(config_path: str) -> AetheristTrainer:
    """Create trainer from configuration file."""
    # This would load from YAML/JSON config in a real implementation
    # For now, return default configuration
    
    generator_config = {
        'latent_dim': 512,
        'style_dim': 512,
        'vit_layers': 8,
        'vit_dim': 512,
        'triplane_resolution': 64,
        'triplane_channels': 32,
    }
    
    discriminator_config = {
        'input_size': 256,
        'input_channels': 3,
        'base_channels': 64,
        'feature_dim': 256,
        'use_multiscale': True,
        'use_consistency_branch': True,
        'num_views': 2,
    }
    
    return AetheristTrainer(
        generator_config=generator_config,
        discriminator_config=discriminator_config,
        batch_size=16,
        learning_rate_g=1e-4,
        learning_rate_d=4e-4,
    )