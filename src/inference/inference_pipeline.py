"""
Inference pipeline for Aetherist GAN.
Handles model loading, image generation, and post-processing.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import json
import logging

from ..models.generator import AetheristGenerator
from ..utils.camera_utils import sample_camera_poses


logger = logging.getLogger(__name__)


class AetheristInferencePipeline:
    """
    Complete inference pipeline for Aetherist GAN.
    
    Supports:
    - Model loading from checkpoints
    - Image generation with various camera poses
    - Batch processing for multiple samples
    - Post-processing and output formatting
    - Latent code interpolation
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        half_precision: bool = False,
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on ("auto", "cuda", "cpu")
            half_precision: Whether to use half precision (FP16) for faster inference
        """
        self.model_path = Path(model_path)
        self.device = self._get_device(device)
        self.half_precision = half_precision and self.device.type == "cuda"
        
        # Model components
        self.generator = None
        self.model_config = None
        
        # Load model
        self._load_model()
        
    def _get_device(self, device: str) -> torch.device:
        """Get the appropriate device for inference."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_model(self):
        """Load the trained model from checkpoint."""
        logger.info(f"Loading model from {self.model_path}")
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model configuration
        if "model_config" in checkpoint:
            self.model_config = checkpoint["model_config"]
        else:
            # Default configuration if not saved
            logger.warning("Model config not found in checkpoint, using defaults")
            self.model_config = {
                "latent_dim": 256,
                "vit_dim": 256,
                "vit_layers": 8,
                "triplane_resolution": 64,
                "triplane_channels": 32,
            }
        
        # Create generator
        self.generator = AetheristGenerator(**self.model_config)
        
        # Load state dict
        if "generator_state_dict" in checkpoint:
            state_dict = checkpoint["generator_state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint  # Assume the whole checkpoint is the state dict
            
        self.generator.load_state_dict(state_dict)
        
        # Move to device and set precision
        self.generator.to(self.device)
        if self.half_precision:
            self.generator.half()
        
        # Set to evaluation mode
        self.generator.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
        logger.info(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
    
    @torch.no_grad()
    def generate(
        self,
        num_samples: int = 1,
        latent_codes: Optional[torch.Tensor] = None,
        camera_poses: Optional[torch.Tensor] = None,
        resolution: Optional[int] = None,
        batch_size: Optional[int] = None,
        return_triplane: bool = False,
        seed: Optional[int] = None,
    ) -> Dict[str, Union[torch.Tensor, List[Image.Image]]]:
        """
        Generate images using the trained model.
        
        Args:
            num_samples: Number of images to generate
            latent_codes: Custom latent codes [N, latent_dim]. If None, random codes are used
            camera_poses: Custom camera poses [N, 4, 4]. If None, random poses are used
            resolution: Output resolution. If None, uses model's default
            batch_size: Batch size for generation. If None, processes all at once
            return_triplane: Whether to return triplane representations
            seed: Random seed for reproducible generation
            
        Returns:
            Dictionary containing:
            - "images": Generated images as PIL Images
            - "images_tensor": Generated images as tensors [N, C, H, W]
            - "latent_codes": Used latent codes [N, latent_dim]
            - "camera_poses": Used camera poses [N, 4, 4]
            - "triplanes": Triplane representations [N, 3, C, H, W] (if requested)
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Prepare latent codes
        if latent_codes is None:
            latent_codes = torch.randn(
                num_samples, 
                self.model_config["latent_dim"],
                device=self.device,
                dtype=torch.float16 if self.half_precision else torch.float32
            )
        else:
            latent_codes = latent_codes.to(self.device)
            if self.half_precision:
                latent_codes = latent_codes.half()
            num_samples = latent_codes.shape[0]
        
        # Prepare camera poses
        if camera_poses is None:
            camera_poses = sample_camera_poses(
                num_samples,
                device=self.device,
                radius_range=(1.5, 2.5),
                elevation_range=(-15, 45),
                azimuth_range=(-180, 180)
            )
        else:
            camera_poses = camera_poses.to(self.device)
            if self.half_precision:
                camera_poses = camera_poses.half()
        
        # Determine batch size
        if batch_size is None:
            batch_size = num_samples
        
        # Generate in batches
        all_images = []
        all_triplanes = [] if return_triplane else None
        
        for i in range(0, num_samples, batch_size):
            end_idx = min(i + batch_size, num_samples)
            batch_latents = latent_codes[i:end_idx]
            batch_poses = camera_poses[i:end_idx]
            
            # Generate batch
            if return_triplane:
                batch_images, batch_triplanes = self.generator(
                    batch_latents, 
                    batch_poses, 
                    return_triplane=True
                )
                all_triplanes.append(batch_triplanes.cpu())
            else:
                batch_images = self.generator(batch_latents, batch_poses)
            
            all_images.append(batch_images.cpu())
        
        # Concatenate results
        images_tensor = torch.cat(all_images, dim=0)
        if return_triplane:
            triplanes_tensor = torch.cat(all_triplanes, dim=0)
        
        # Post-process images
        images_pil = self._tensor_to_pil(images_tensor, resolution)
        
        # Prepare output
        result = {
            "images": images_pil,
            "images_tensor": images_tensor,
            "latent_codes": latent_codes.cpu(),
            "camera_poses": camera_poses.cpu(),
        }
        
        if return_triplane:
            result["triplanes"] = triplanes_tensor
        
        return result
    
    def interpolate(
        self,
        start_latent: torch.Tensor,
        end_latent: torch.Tensor,
        steps: int = 10,
        camera_pose: Optional[torch.Tensor] = None,
        interpolation_method: str = "slerp",
    ) -> Dict[str, Union[torch.Tensor, List[Image.Image]]]:
        """
        Generate interpolated images between two latent codes.
        
        Args:
            start_latent: Starting latent code [latent_dim]
            end_latent: Ending latent code [latent_dim]
            steps: Number of interpolation steps
            camera_pose: Fixed camera pose for all frames [4, 4]
            interpolation_method: "linear" or "slerp" (spherical linear interpolation)
            
        Returns:
            Dictionary containing interpolated images and latent codes
        """
        # Prepare latent codes
        start_latent = start_latent.to(self.device).unsqueeze(0)
        end_latent = end_latent.to(self.device).unsqueeze(0)
        
        if self.half_precision:
            start_latent = start_latent.half()
            end_latent = end_latent.half()
        
        # Generate interpolated latent codes
        if interpolation_method == "linear":
            alphas = torch.linspace(0, 1, steps, device=self.device)
            interpolated_latents = []
            for alpha in alphas:
                latent = (1 - alpha) * start_latent + alpha * end_latent
                interpolated_latents.append(latent)
            interpolated_latents = torch.cat(interpolated_latents, dim=0)
        
        elif interpolation_method == "slerp":
            # Spherical linear interpolation
            def slerp(v1, v2, t):
                v1_norm = F.normalize(v1, dim=-1)
                v2_norm = F.normalize(v2, dim=-1)
                dot = torch.sum(v1_norm * v2_norm, dim=-1, keepdim=True)
                dot = torch.clamp(dot, -1.0, 1.0)
                
                omega = torch.acos(torch.abs(dot))
                sin_omega = torch.sin(omega)
                
                # Avoid division by zero
                sin_omega = torch.where(sin_omega < 1e-6, torch.ones_like(sin_omega), sin_omega)
                
                coeff1 = torch.sin((1.0 - t) * omega) / sin_omega
                coeff2 = torch.sin(t * omega) / sin_omega
                
                return coeff1 * v1 + coeff2 * v2
            
            alphas = torch.linspace(0, 1, steps, device=self.device)
            interpolated_latents = []
            for alpha in alphas:
                latent = slerp(start_latent, end_latent, alpha)
                interpolated_latents.append(latent)
            interpolated_latents = torch.cat(interpolated_latents, dim=0)
        
        else:
            raise ValueError(f"Unknown interpolation method: {interpolation_method}")
        
        # Use fixed camera pose or sample random ones
        if camera_pose is not None:
            camera_poses = camera_pose.unsqueeze(0).repeat(steps, 1, 1).to(self.device)
        else:
            camera_poses = None
        
        # Generate interpolated images
        return self.generate(
            num_samples=steps,
            latent_codes=interpolated_latents,
            camera_poses=camera_poses,
        )
    
    def _tensor_to_pil(
        self, 
        tensor: torch.Tensor, 
        resolution: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Convert tensor to PIL images with proper post-processing.
        
        Args:
            tensor: Image tensor [N, C, H, W] in range [-1, 1]
            resolution: Target resolution for resizing
            
        Returns:
            List of PIL Images
        """
        # Convert to [0, 1] range
        tensor = (tensor + 1) / 2
        tensor = torch.clamp(tensor, 0, 1)
        
        # Resize if needed
        if resolution is not None and tensor.shape[-1] != resolution:
            tensor = F.interpolate(
                tensor, 
                size=(resolution, resolution), 
                mode="bilinear", 
                align_corners=False
            )
        
        # Convert to numpy and PIL
        tensor_np = tensor.permute(0, 2, 3, 1).cpu().numpy()
        tensor_np = (tensor_np * 255).astype(np.uint8)
        
        images = []
        for i in range(tensor_np.shape[0]):
            img = Image.fromarray(tensor_np[i])
            images.append(img)
        
        return images
    
    def save_images(
        self,
        images: List[Image.Image],
        output_dir: str,
        prefix: str = "generated",
        format: str = "PNG",
    ) -> List[str]:
        """
        Save generated images to disk.
        
        Args:
            images: List of PIL images to save
            output_dir: Output directory
            prefix: Filename prefix
            format: Image format ("PNG", "JPEG", etc.)
            
        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        for i, img in enumerate(images):
            filename = f"{prefix}_{i:04d}.{format.lower()}"
            filepath = output_dir / filename
            img.save(filepath, format=format)
            saved_paths.append(str(filepath))
        
        return saved_paths
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            "model_path": str(self.model_path),
            "device": str(self.device),
            "half_precision": self.half_precision,
            "model_config": self.model_config,
            "parameter_count": sum(p.numel() for p in self.generator.parameters()),
            "memory_usage_mb": sum(p.numel() * p.element_size() for p in self.generator.parameters()) / 1024**2,
        }


class BatchInferencePipeline:
    """
    Optimized pipeline for large batch inference.
    Includes memory management and progress tracking.
    """
    
    def __init__(
        self,
        inference_pipeline: AetheristInferencePipeline,
        max_batch_size: int = 8,
        enable_progress: bool = True,
    ):
        """
        Initialize batch inference pipeline.
        
        Args:
            inference_pipeline: Base inference pipeline
            max_batch_size: Maximum batch size to prevent OOM
            enable_progress: Whether to show progress bars
        """
        self.pipeline = inference_pipeline
        self.max_batch_size = max_batch_size
        self.enable_progress = enable_progress
    
    def generate_large_batch(
        self,
        num_samples: int,
        output_dir: str,
        **generate_kwargs
    ) -> Dict:
        """
        Generate a large number of samples efficiently.
        
        Args:
            num_samples: Total number of samples to generate
            output_dir: Directory to save images
            **generate_kwargs: Arguments passed to generate()
            
        Returns:
            Dictionary with generation statistics
        """
        import time
        from tqdm import tqdm if self.enable_progress else None
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        total_batches = (num_samples + self.max_batch_size - 1) // self.max_batch_size
        
        if self.enable_progress and tqdm:
            pbar = tqdm(total=num_samples, desc="Generating images")
        
        start_time = time.time()
        saved_paths = []
        
        for batch_idx in range(total_batches):
            batch_start = batch_idx * self.max_batch_size
            batch_end = min(batch_start + self.max_batch_size, num_samples)
            batch_size = batch_end - batch_start
            
            # Generate batch
            result = self.pipeline.generate(
                num_samples=batch_size,
                batch_size=batch_size,
                **generate_kwargs
            )
            
            # Save images
            batch_paths = self.pipeline.save_images(
                result["images"],
                output_dir,
                prefix=f"batch_{batch_idx:03d}_sample"
            )
            saved_paths.extend(batch_paths)
            
            if self.enable_progress and tqdm:
                pbar.update(batch_size)
            
            # Clear cache to prevent memory issues
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        if self.enable_progress and tqdm:
            pbar.close()
        
        end_time = time.time()
        
        return {
            "num_generated": num_samples,
            "total_time": end_time - start_time,
            "avg_time_per_sample": (end_time - start_time) / num_samples,
            "saved_paths": saved_paths,
        }