"""
Training loop implementation for Aetherist.

This module implements the comprehensive training system for the Aetherist GAN,
including loss functions, optimizers, learning rate scheduling, and training orchestration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Any
import math
import time
from pathlib import Path

from ..models.generator import AetheristGenerator
from ..models.discriminator import AetheristDiscriminator
from ..utils.camera import sample_camera_poses, perspective_projection_matrix


class AdversarialLoss(nn.Module):
    """
    Adversarial loss implementation with multiple loss types.
    
    Supports various GAN loss formulations including:
    - Non-saturating GAN loss
    - Wasserstein GAN loss
    - R1 gradient penalty
    - Progressive growing losses
    """
    
    def __init__(
        self,
        loss_type: str = "non_saturating",
        r1_gamma: float = 10.0,
        use_gradient_penalty: bool = True,
    ):
        """
        Initialize adversarial loss.
        
        Args:
            loss_type: Type of adversarial loss ('non_saturating', 'wasserstein', 'hinge')
            r1_gamma: R1 gradient penalty weight
            use_gradient_penalty: Whether to use gradient penalty
        """
        super().__init__()
        
        self.loss_type = loss_type
        self.r1_gamma = r1_gamma
        self.use_gradient_penalty = use_gradient_penalty
        
        if loss_type not in ["non_saturating", "wasserstein", "hinge"]:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def generator_loss(self, fake_logits: Tensor) -> Tensor:
        """
        Compute generator loss.
        
        Args:
            fake_logits: Discriminator outputs for fake images
            
        Returns:
            Generator loss tensor
        """
        if self.loss_type == "non_saturating":
            # -log(D(G(z)))
            return F.softplus(-fake_logits).mean()
        elif self.loss_type == "wasserstein":
            # -D(G(z))
            return -fake_logits.mean()
        elif self.loss_type == "hinge":
            # -D(G(z))
            return -fake_logits.mean()
        
    def discriminator_loss(
        self,
        real_logits: Tensor,
        fake_logits: Tensor,
    ) -> Tensor:
        """
        Compute discriminator loss.
        
        Args:
            real_logits: Discriminator outputs for real images
            fake_logits: Discriminator outputs for fake images
            
        Returns:
            Discriminator loss tensor
        """
        if self.loss_type == "non_saturating":
            # -log(D(x)) - log(1 - D(G(z)))
            real_loss = F.softplus(-real_logits).mean()
            fake_loss = F.softplus(fake_logits).mean()
            return real_loss + fake_loss
        elif self.loss_type == "wasserstein":
            # -D(x) + D(G(z))
            return -real_logits.mean() + fake_logits.mean()
        elif self.loss_type == "hinge":
            # -min(0, -1 + D(x)) - min(0, -1 - D(G(z)))
            real_loss = F.relu(1.0 - real_logits).mean()
            fake_loss = F.relu(1.0 + fake_logits).mean()
            return real_loss + fake_loss
    
    def r1_penalty(
        self,
        real_images: Tensor,
        real_logits: Tensor,
    ) -> Tensor:
        """
        Compute R1 gradient penalty.
        
        Args:
            real_images: Real input images
            real_logits: Discriminator outputs for real images
            
        Returns:
            R1 penalty tensor
        """
        # Compute gradients w.r.t. real images
        gradients = torch.autograd.grad(
            outputs=real_logits.sum(),
            inputs=real_images,
            create_graph=True,
            retain_graph=True,
        )[0]
        
        # Compute penalty: ||∇D(x)||²
        penalty = gradients.pow(2).sum(dim=[1, 2, 3]).mean()
        return self.r1_gamma * 0.5 * penalty


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using pre-trained VGG features.
    
    Computes feature-space distance between generated and target images
    using multiple layers of a pre-trained VGG network.
    """
    
    def __init__(
        self,
        layers: List[str] = None,
        weights: List[float] = None,
        normalize: bool = True,
    ):
        """
        Initialize perceptual loss.
        
        Args:
            layers: VGG layer names to use for loss computation
            weights: Weights for each layer's contribution
            normalize: Whether to normalize VGG inputs
        """
        super().__init__()
        
        if layers is None:
            layers = ['relu1_2', 'relu2_2', 'relu3_2', 'relu4_2']
        
        if weights is None:
            weights = [1.0] * len(layers)
        
        self.layers = layers
        self.weights = weights
        self.normalize = normalize
        
        # Load pre-trained VGG
        try:
            import torchvision.models as models
            vgg = models.vgg19(pretrained=True).features
            self.vgg = vgg.eval()
            
            # Freeze parameters
            for param in self.vgg.parameters():
                param.requires_grad = False
                
        except ImportError:
            print("Warning: torchvision not available, perceptual loss disabled")
            self.vgg = None
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute perceptual loss.
        
        Args:
            pred: Predicted images (B, C, H, W)
            target: Target images (B, C, H, W)
            
        Returns:
            Perceptual loss tensor
        """
        if self.vgg is None:
            return torch.tensor(0.0, device=pred.device)
        
        # Normalize inputs to [0, 1] range
        if self.normalize:
            pred = (pred + 1) / 2
            target = (target + 1) / 2
        
        # Extract features from both images
        pred_features = self._extract_features(pred)
        target_features = self._extract_features(target)
        
        # Compute weighted loss across layers
        loss = 0.0
        for layer, weight in zip(self.layers, self.weights):
            if layer in pred_features and layer in target_features:
                layer_loss = F.mse_loss(pred_features[layer], target_features[layer])
                loss += weight * layer_loss
        
        return loss
    
    def _extract_features(self, x: Tensor) -> Dict[str, Tensor]:
        """Extract features from VGG layers."""
        features = {}
        
        layer_names = {
            '3': 'relu1_2', '8': 'relu2_2', '13': 'relu3_2', 
            '22': 'relu4_2', '31': 'relu5_2'
        }
        
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if str(i) in layer_names:
                features[layer_names[str(i)]] = x
        
        return features


class ConsistencyLoss(nn.Module):
    """
    3D consistency loss for multi-view generation.
    
    Ensures that generated images from different viewpoints maintain
    geometric consistency and 3D structure coherence.
    """
    
    def __init__(
        self,
        lambda_consistency: float = 1.0,
        use_depth_consistency: bool = True,
        use_normal_consistency: bool = True,
    ):
        """
        Initialize consistency loss.
        
        Args:
            lambda_consistency: Weight for consistency loss
            use_depth_consistency: Whether to enforce depth consistency
            use_normal_consistency: Whether to enforce normal consistency
        """
        super().__init__()
        
        self.lambda_consistency = lambda_consistency
        self.use_depth_consistency = use_depth_consistency
        self.use_normal_consistency = use_normal_consistency
    
    def forward(
        self,
        view1_output: Dict[str, Tensor],
        view2_output: Dict[str, Tensor],
        camera1: Tensor,
        camera2: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Compute consistency loss between two views.
        
        Args:
            view1_output: Generator output for view 1
            view2_output: Generator output for view 2
            camera1: Camera matrix for view 1
            camera2: Camera matrix for view 2
            
        Returns:
            Dictionary containing consistency losses
        """
        losses = {}
        
        # Image-space consistency (photometric)
        if 'high_res_image' in view1_output and 'high_res_image' in view2_output:
            # For now, use simple MSE between overlapping regions
            # In practice, this would involve warping and computing overlap
            img_consistency = F.mse_loss(
                view1_output['high_res_image'].mean(dim=[2, 3]),
                view2_output['high_res_image'].mean(dim=[2, 3])
            )
            losses['image_consistency'] = self.lambda_consistency * img_consistency
        
        # Tri-plane consistency
        if 'triplanes' in view1_output and 'triplanes' in view2_output:
            triplane_consistency = 0.0
            for plane_name in view1_output['triplanes']:
                plane1 = view1_output['triplanes'][plane_name]
                plane2 = view2_output['triplanes'][plane_name]
                triplane_consistency += F.mse_loss(plane1, plane2)
            
            losses['triplane_consistency'] = self.lambda_consistency * triplane_consistency / 3
        
        # Feature consistency
        if 'vit_features' in view1_output and 'vit_features' in view2_output:
            feature_consistency = F.cosine_embedding_loss(
                view1_output['vit_features'].flatten(1),
                view2_output['vit_features'].flatten(1),
                torch.ones(view1_output['vit_features'].size(0), device=view1_output['vit_features'].device)
            )
            losses['feature_consistency'] = self.lambda_consistency * feature_consistency
        
        return losses


class TrainingScheduler:
    """
    Learning rate and training scheduler for Aetherist.
    
    Implements progressive training strategies, learning rate scheduling,
    and training phase management.
    """
    
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
        generator_lr: float = 2e-4,
        discriminator_lr: float = 2e-4,
        beta1: float = 0.0,
        beta2: float = 0.999,
        scheduler_type: str = "cosine",
        warmup_steps: int = 1000,
        total_steps: int = 100000,
    ):
        """
        Initialize training scheduler.
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            generator_lr: Initial learning rate for generator
            discriminator_lr: Initial learning rate for discriminator
            beta1: Adam beta1 parameter
            beta2: Adam beta2 parameter
            scheduler_type: Learning rate scheduler type
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
        """
        self.generator = generator
        self.discriminator = discriminator
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        
        # Initialize optimizers
        self.optimizer_G = torch.optim.Adam(
            generator.parameters(),
            lr=generator_lr,
            betas=(beta1, beta2),
            eps=1e-8,
        )
        
        self.optimizer_D = torch.optim.Adam(
            discriminator.parameters(),
            lr=discriminator_lr,
            betas=(beta1, beta2),
            eps=1e-8,
        )
        
        # Initialize schedulers
        if scheduler_type == "cosine":
            self.scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_G, T_max=total_steps
            )
            self.scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer_D, T_max=total_steps
            )
        elif scheduler_type == "linear":
            lambda_fn = lambda step: max(0, 1 - step / total_steps)
            self.scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimizer_G, lambda_fn)
            self.scheduler_D = torch.optim.lr_scheduler.LambdaLR(self.optimizer_D, lambda_fn)
        else:
            self.scheduler_G = None
            self.scheduler_D = None
        
        self.current_step = 0
    
    def step(self):
        """Perform scheduler step."""
        self.current_step += 1
        
        # Warmup phase
        if self.current_step <= self.warmup_steps:
            warmup_factor = self.current_step / self.warmup_steps
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = param_group['initial_lr'] * warmup_factor
            for param_group in self.optimizer_D.param_groups:
                param_group['lr'] = param_group['initial_lr'] * warmup_factor
        else:
            # Normal scheduling
            if self.scheduler_G is not None:
                self.scheduler_G.step()
            if self.scheduler_D is not None:
                self.scheduler_D.step()
    
    def get_lr(self) -> Dict[str, float]:
        """Get current learning rates."""
        return {
            'generator': self.optimizer_G.param_groups[0]['lr'],
            'discriminator': self.optimizer_D.param_groups[0]['lr'],
        }


class AetheristTrainer:
    """
    Main trainer class for Aetherist.
    
    Orchestrates the complete training process including:
    - Loss computation and backpropagation
    - Multi-view consistency training
    - Progressive training strategies
    - Logging and checkpointing
    """
    
    def __init__(
        self,
        generator: AetheristGenerator,
        discriminator: AetheristDiscriminator,
        device: torch.device = None,
        
        # Loss configuration
        adversarial_loss_type: str = "non_saturating",
        lambda_perceptual: float = 1.0,
        lambda_consistency: float = 0.1,
        lambda_r1: float = 10.0,
        
        # Training configuration
        generator_lr: float = 2e-4,
        discriminator_lr: float = 2e-4,
        d_reg_every: int = 16,
        g_reg_every: int = 4,
        
        # Multi-view configuration
        num_views: int = 2,
        view_probability: float = 0.5,
        
        # Logging configuration
        log_every: int = 100,
        save_every: int = 5000,
        sample_every: int = 1000,
    ):
        """
        Initialize Aetherist trainer.
        
        Args:
            generator: Generator model
            discriminator: Discriminator model
            device: Training device
            adversarial_loss_type: Type of adversarial loss
            lambda_perceptual: Weight for perceptual loss
            lambda_consistency: Weight for consistency loss
            lambda_r1: Weight for R1 gradient penalty
            generator_lr: Generator learning rate
            discriminator_lr: Discriminator learning rate
            d_reg_every: Discriminator regularization frequency
            g_reg_every: Generator regularization frequency
            num_views: Number of views for consistency training
            view_probability: Probability of using multi-view training
            log_every: Logging frequency
            save_every: Checkpoint saving frequency
            sample_every: Sample generation frequency
        """
        self.generator = generator
        self.discriminator = discriminator
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move models to device
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Initialize loss functions
        self.adversarial_loss = AdversarialLoss(
            loss_type=adversarial_loss_type,
            r1_gamma=lambda_r1,
            use_gradient_penalty=True,
        )
        self.perceptual_loss = PerceptualLoss()
        self.consistency_loss = ConsistencyLoss(lambda_consistency=lambda_consistency)
        
        # Loss weights
        self.lambda_perceptual = lambda_perceptual
        self.lambda_consistency = lambda_consistency
        
        # Training configuration
        self.d_reg_every = d_reg_every
        self.g_reg_every = g_reg_every
        self.num_views = num_views
        self.view_probability = view_probability
        
        # Logging configuration
        self.log_every = log_every
        self.save_every = save_every
        self.sample_every = sample_every
        
        # Initialize scheduler
        self.scheduler = TrainingScheduler(
            generator=generator,
            discriminator=discriminator,
            generator_lr=generator_lr,
            discriminator_lr=discriminator_lr,
        )
        
        # Training state
        self.step = 0
        self.epoch = 0
        
        # Metrics tracking
        self.metrics_history = {
            'generator_loss': [],
            'discriminator_loss': [],
            'perceptual_loss': [],
            'consistency_loss': [],
            'r1_penalty': [],
        }
    
    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            batch: Training batch containing images and camera parameters
            
        Returns:
            Dictionary of loss values
        """
        batch_size = batch['images'].size(0)
        real_images = batch['images'].to(self.device)
        
        # Sample latent codes
        z = torch.randn(batch_size, self.generator.latent_dim, device=self.device)
        
        # Sample camera poses for multi-view training
        if torch.rand(1).item() < self.view_probability and self.num_views > 1:
            # Multi-view training
            cameras, view_angles = self._sample_multi_view_cameras(batch_size)
            use_multi_view = True
        else:
            # Single view training
            cameras, view_angles = self._sample_single_view_camera(batch_size)
            use_multi_view = False
        
        # -------------------
        # Train Discriminator
        # -------------------
        self.discriminator.zero_grad()
        
        # Generate fake images
        with torch.no_grad():
            if use_multi_view:
                fake_outputs = []
                for cam in cameras:
                    fake_output = self.generator(z, cam, return_triplanes=True)
                    fake_outputs.append(fake_output)
                fake_images = [output['high_res_image'] for output in fake_outputs]
            else:
                fake_output = self.generator(z, cameras[0], return_triplanes=True)
                fake_images = [fake_output['high_res_image']]
        
        # Discriminator forward pass
        real_images.requires_grad_(True)
        real_logits = self.discriminator([real_images[0:1]])['final_score']
        fake_logits = self.discriminator(fake_images)['final_score']
        
        # Compute discriminator loss
        d_loss = self.adversarial_loss.discriminator_loss(real_logits, fake_logits)
        
        # R1 regularization (every d_reg_every steps)
        r1_penalty = torch.tensor(0.0, device=self.device)
        if self.step % self.d_reg_every == 0:
            r1_penalty = self.adversarial_loss.r1_penalty(real_images, real_logits)
            d_loss = d_loss + r1_penalty
        
        # Backward pass
        d_loss.backward()
        self.scheduler.optimizer_D.step()
        
        # ---------------
        # Train Generator
        # ---------------
        self.generator.zero_grad()
        
        # Generate fake images (with gradients)
        if use_multi_view:
            fake_outputs = []
            for cam in cameras:
                fake_output = self.generator(z, cam, return_triplanes=True)
                fake_outputs.append(fake_output)
            fake_images = [output['high_res_image'] for output in fake_outputs]
        else:
            fake_output = self.generator(z, cameras[0], return_triplanes=True)
            fake_outputs = [fake_output]
            fake_images = [fake_output['high_res_image']]
        
        # Discriminator evaluation
        fake_logits = self.discriminator(fake_images)['final_score']
        
        # Adversarial loss
        g_adv_loss = self.adversarial_loss.generator_loss(fake_logits)
        
        # Perceptual loss (compare with real images)
        perceptual_loss = torch.tensor(0.0, device=self.device)
        if self.lambda_perceptual > 0:
            for fake_img in fake_images:
                # Use a random real image for perceptual comparison
                real_idx = torch.randint(0, real_images.size(0), (1,)).item()
                perceptual_loss += self.perceptual_loss(fake_img, real_images[real_idx:real_idx+1])
            perceptual_loss = perceptual_loss / len(fake_images)
        
        # Consistency loss (multi-view only)
        consistency_loss = torch.tensor(0.0, device=self.device)
        if use_multi_view and self.lambda_consistency > 0:
            consistency_losses = self.consistency_loss(
                fake_outputs[0], fake_outputs[1], cameras[0], cameras[1]
            )
            consistency_loss = sum(consistency_losses.values())
        
        # Total generator loss
        g_loss = g_adv_loss + self.lambda_perceptual * perceptual_loss + consistency_loss
        
        # Backward pass
        g_loss.backward()
        self.scheduler.optimizer_G.step()
        
        # Update scheduler
        self.scheduler.step()
        
        # Return metrics
        metrics = {
            'generator_loss': g_loss.item(),
            'generator_adv_loss': g_adv_loss.item(),
            'discriminator_loss': d_loss.item(),
            'perceptual_loss': perceptual_loss.item(),
            'consistency_loss': consistency_loss.item(),
            'r1_penalty': r1_penalty.item(),
            'lr_g': self.scheduler.get_lr()['generator'],
            'lr_d': self.scheduler.get_lr()['discriminator'],
        }
        
        # Update metrics history
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)
        
        self.step += 1
        return metrics
    
    def _sample_multi_view_cameras(self, batch_size: int) -> Tuple[List[Tensor], List[Tensor]]:
        """Sample multiple camera viewpoints for consistency training."""
        cameras = []
        view_angles = []
        
        for _ in range(self.num_views):
            eye_pos, view_matrix, angles = sample_camera_poses(batch_size, device=self.device)
            proj_matrix = perspective_projection_matrix(
                fov_degrees=torch.tensor([50.0] * batch_size),
                aspect_ratio=torch.tensor([1.0] * batch_size),
            ).to(self.device)
            camera_matrix = torch.bmm(proj_matrix, view_matrix)
            
            cameras.append(camera_matrix)
            view_angles.append(angles)
        
        return cameras, view_angles
    
    def _sample_single_view_camera(self, batch_size: int) -> Tuple[List[Tensor], List[Tensor]]:
        """Sample single camera viewpoint."""
        eye_pos, view_matrix, angles = sample_camera_poses(batch_size, device=self.device)
        proj_matrix = perspective_projection_matrix(
            fov_degrees=torch.tensor([50.0] * batch_size),
            aspect_ratio=torch.tensor([1.0] * batch_size),
        ).to(self.device)
        camera_matrix = torch.bmm(proj_matrix, view_matrix)
        
        return [camera_matrix], [angles]
    
    def save_checkpoint(self, filepath: str):
        """Save training checkpoint."""
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_G_state_dict': self.scheduler.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.scheduler.optimizer_D.state_dict(),
            'metrics_history': self.metrics_history,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.scheduler.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.scheduler.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        self.metrics_history = checkpoint['metrics_history']
        
        print(f"Checkpoint loaded from {filepath}")
    
    def generate_samples(self, num_samples: int = 8, save_path: Optional[str] = None) -> List[Tensor]:
        """Generate sample images for evaluation."""
        self.generator.eval()
        
        with torch.no_grad():
            # Sample latent codes
            z = torch.randn(num_samples, self.generator.latent_dim, device=self.device)
            
            # Sample camera poses
            eye_pos, view_matrix, angles = sample_camera_poses(num_samples, device=self.device)
            proj_matrix = perspective_projection_matrix(
                fov_degrees=torch.tensor([50.0] * num_samples),
                aspect_ratio=torch.tensor([1.0] * num_samples),
            ).to(self.device)
            camera_matrix = torch.bmm(proj_matrix, view_matrix)
            
            # Generate images
            output = self.generator(z, camera_matrix)
            samples = output['high_res_image']
            
            # Save samples if path provided
            if save_path is not None:
                # Convert to [0, 1] range and save
                samples_vis = torch.clamp((samples + 1) / 2, 0, 1)
                
                try:
                    import torchvision.utils as vutils
                    vutils.save_image(
                        samples_vis,
                        save_path,
                        nrow=int(math.sqrt(num_samples)),
                        padding=2,
                        normalize=False,
                    )
                    print(f"Samples saved to {save_path}")
                except ImportError:
                    print("torchvision not available, samples not saved")
        
        self.generator.train()
        return samples


def test_training_components():
    """Test function for training components."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 2
    latent_dim = 512
    img_size = 256
    
    print("Testing Aetherist Training Components...")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print()
    
    # Create models
    from ..models.generator import AetheristGenerator
    from ..models.discriminator import AetheristDiscriminator
    
    generator = AetheristGenerator(
        vit_dim=256,
        vit_layers=4,
        triplane_resolution=64,
        triplane_channels=32,
    ).to(device)
    
    discriminator = AetheristDiscriminator(
        input_size=img_size,
        use_consistency_branch=True,
        num_views=2,
    ).to(device)
    
    print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
    print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")
    print()
    
    # Test loss functions
    print("Testing Loss Functions...")
    
    # Test adversarial loss
    adv_loss = AdversarialLoss()
    fake_logits = torch.randn(batch_size, 1, device=device)
    real_logits = torch.randn(batch_size, 1, device=device)
    
    g_loss = adv_loss.generator_loss(fake_logits)
    d_loss = adv_loss.discriminator_loss(real_logits, fake_logits)
    
    print(f"Generator loss: {g_loss.item():.4f}")
    print(f"Discriminator loss: {d_loss.item():.4f}")
    
    # Test perceptual loss
    perceptual_loss = PerceptualLoss()
    if perceptual_loss.vgg is not None:
        fake_images = torch.randn(batch_size, 3, img_size, img_size, device=device)
        real_images = torch.randn(batch_size, 3, img_size, img_size, device=device)
        p_loss = perceptual_loss(fake_images, real_images)
        print(f"Perceptual loss: {p_loss.item():.4f}")
    else:
        print("Perceptual loss: Not available (no torchvision)")
    
    print()
    
    # Test trainer
    print("Testing Trainer...")
    trainer = AetheristTrainer(
        generator=generator,
        discriminator=discriminator,
        device=device,
        lambda_perceptual=0.1,
        lambda_consistency=0.05,
    )
    
    # Create dummy batch
    batch = {
        'images': torch.randn(batch_size, 3, img_size, img_size, device=device)
    }
    
    # Test training step
    metrics = trainer.train_step(batch)
    
    print("Training metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")
    print()
    
    # Test sample generation
    print("Testing Sample Generation...")
    samples = trainer.generate_samples(num_samples=4)
    print(f"Generated samples shape: {samples.shape}")
    
    print("✅ All training component tests passed!")


if __name__ == "__main__":
    test_training_components()