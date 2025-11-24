"""
Camera utilities for Aetherist 3D-aware generation.
Handles 3D camera mathematics, pose sampling, and ray generation for NeRF rendering.
"""

import math
import random
from typing import Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


def look_at_matrix(
    eye: Tensor,
    target: Tensor,
    up: Tensor,
) -> Tensor:
    """
    Create a look-at view matrix.
    
    The look-at matrix transforms world coordinates to camera coordinates.
    Mathematical formulation:
    - forward = normalize(target - eye)
    - right = normalize(cross(forward, up))
    - up = cross(right, forward)
    
    Args:
        eye: Camera position (B, 3) or (3,)
        target: Target position (B, 3) or (3,)  
        up: Up vector (B, 3) or (3,)
        
    Returns:
        View matrix (B, 4, 4) or (4, 4)
    """
    # Ensure we're working with batched tensors
    if eye.dim() == 1:
        eye = eye.unsqueeze(0)
        target = target.unsqueeze(0)
        up = up.unsqueeze(0)
        squeeze_batch = True
    else:
        squeeze_batch = False
    
    batch_size = eye.size(0)
    device = eye.device
    
    # Compute camera coordinate system
    forward = F.normalize(target - eye, dim=-1)
    right = F.normalize(torch.cross(forward, up, dim=-1), dim=-1)
    up = torch.cross(right, forward, dim=-1)
    
    # Create rotation matrix (transpose because we want world-to-camera)
    rotation = torch.stack([right, up, -forward], dim=-1)  # (B, 3, 3)
    
    # Create translation vector
    translation = -torch.bmm(rotation, eye.unsqueeze(-1)).squeeze(-1)  # (B, 3)
    
    # Construct 4x4 transformation matrix
    view_matrix = torch.zeros(batch_size, 4, 4, device=device, dtype=eye.dtype)
    view_matrix[:, :3, :3] = rotation
    view_matrix[:, :3, 3] = translation
    view_matrix[:, 3, 3] = 1.0
    
    if squeeze_batch:
        view_matrix = view_matrix.squeeze(0)
    
    return view_matrix


def perspective_projection_matrix(
    fov_degrees: Union[float, Tensor],
    aspect_ratio: Union[float, Tensor],
    near: Union[float, Tensor] = 0.1,
    far: Union[float, Tensor] = 100.0,
) -> Tensor:
    """
    Create perspective projection matrix.
    
    Standard perspective projection using field of view.
    Mathematical formulation:
    - f = 1 / tan(fov/2)
    - P[0,0] = f / aspect_ratio
    - P[1,1] = f
    - P[2,2] = (far + near) / (near - far)  
    - P[2,3] = (2 * far * near) / (near - far)
    - P[3,2] = -1
    
    Args:
        fov_degrees: Field of view in degrees
        aspect_ratio: Width / height ratio
        near: Near clipping plane
        far: Far clipping plane
        
    Returns:
        Projection matrix (4, 4) or (B, 4, 4)
    """
    # Convert to tensors if needed
    if not isinstance(fov_degrees, Tensor):
        fov_degrees = torch.tensor(fov_degrees, dtype=torch.float32)
    if not isinstance(aspect_ratio, Tensor):
        aspect_ratio = torch.tensor(aspect_ratio, dtype=torch.float32)
    if not isinstance(near, Tensor):
        near = torch.tensor(near, dtype=torch.float32)
    if not isinstance(far, Tensor):
        far = torch.tensor(far, dtype=torch.float32)
    
    device = fov_degrees.device
    
    # Handle batch dimension
    if fov_degrees.dim() == 0:
        batch_size = 1
        squeeze_batch = True
        fov_degrees = fov_degrees.unsqueeze(0)
        aspect_ratio = aspect_ratio.expand(1)
        near = near.expand(1)
        far = far.expand(1)
    else:
        batch_size = fov_degrees.size(0)
        squeeze_batch = False
    
    # Convert FOV to radians and compute focal length
    fov_rad = fov_degrees * math.pi / 180.0
    f = 1.0 / torch.tan(fov_rad / 2.0)
    
    # Create projection matrix
    proj_matrix = torch.zeros(batch_size, 4, 4, device=device, dtype=torch.float32)
    proj_matrix[:, 0, 0] = f / aspect_ratio
    proj_matrix[:, 1, 1] = f
    proj_matrix[:, 2, 2] = (far + near) / (near - far)
    proj_matrix[:, 2, 3] = (2 * far * near) / (near - far)
    proj_matrix[:, 3, 2] = -1.0
    
    if squeeze_batch:
        proj_matrix = proj_matrix.squeeze(0)
    
    return proj_matrix


def sample_camera_poses(
    batch_size: int,
    radius: Union[float, Tuple[float, float]] = 1.2,
    elevation_range: Tuple[float, float] = (-15, 15),
    azimuth_range: Tuple[float, float] = (0, 360),
    target: Optional[Tensor] = None,
    up: Optional[Tensor] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Sample random camera poses for training.
    
    Args:
        batch_size: Number of camera poses to sample
        radius: Camera distance from target (single value or (min, max) range)
        elevation_range: Elevation angle range in degrees (vertical rotation)
        azimuth_range: Azimuth angle range in degrees (horizontal rotation)
        target: Target position (defaults to origin)
        up: Up vector (defaults to world up)
        device: Device to create tensors on
        
    Returns:
        Tuple of (eye_positions, view_matrices, camera_angles)
        - eye_positions: Camera positions (B, 3)
        - view_matrices: View transformation matrices (B, 4, 4)
        - camera_angles: [elevation, azimuth, radius] (B, 3)
    """
    if target is None:
        target = torch.zeros(1, 3, device=device)
    if up is None:
        up = torch.tensor([[0, 1, 0]], device=device, dtype=torch.float32)
    
    # Sample camera parameters
    if isinstance(radius, (tuple, list)):
        radius_samples = torch.rand(batch_size, 1, device=device) * (radius[1] - radius[0]) + radius[0]
    else:
        radius_samples = torch.full((batch_size, 1), radius, device=device)
    
    elevation_samples = (
        torch.rand(batch_size, 1, device=device) * (elevation_range[1] - elevation_range[0]) 
        + elevation_range[0]
    )
    azimuth_samples = (
        torch.rand(batch_size, 1, device=device) * (azimuth_range[1] - azimuth_range[0])
        + azimuth_range[0]
    )
    
    # Convert to radians
    elevation_rad = elevation_samples * math.pi / 180.0
    azimuth_rad = azimuth_samples * math.pi / 180.0
    
    # Convert spherical coordinates to Cartesian
    # x = r * cos(elevation) * cos(azimuth)
    # y = r * sin(elevation)  
    # z = r * cos(elevation) * sin(azimuth)
    x = radius_samples * torch.cos(elevation_rad) * torch.cos(azimuth_rad)
    y = radius_samples * torch.sin(elevation_rad)
    z = radius_samples * torch.cos(elevation_rad) * torch.sin(azimuth_rad)
    
    eye_positions = torch.cat([x, y, z], dim=1)  # (B, 3)
    
    # Expand target and up for batch processing
    target_batch = target.expand(batch_size, -1)
    up_batch = up.expand(batch_size, -1)
    
    # Create view matrices
    view_matrices = look_at_matrix(eye_positions, target_batch, up_batch)
    
    # Store camera angles for conditioning/loss
    camera_angles = torch.cat([elevation_samples, azimuth_samples, radius_samples], dim=1)
    
    return eye_positions, view_matrices, camera_angles


def generate_rays(
    camera_matrix: Tensor,
    image_height: int,
    image_width: int,
    fov_degrees: float = 50.0,
    near: float = 0.5,
    far: float = 2.5,
) -> Tuple[Tensor, Tensor]:
    """
    Generate rays for volumetric rendering.
    
    For each pixel, we generate a ray from the camera center through the pixel.
    Ray equation: r(t) = origin + t * direction
    
    Args:
        camera_matrix: Camera-to-world transformation matrix (B, 4, 4)
        image_height: Output image height
        image_width: Output image width
        fov_degrees: Camera field of view in degrees
        near: Near clipping distance
        far: Far clipping distance
        
    Returns:
        Tuple of (ray_origins, ray_directions)
        - ray_origins: Ray starting points (B, H, W, 3)
        - ray_directions: Ray directions (B, H, W, 3)
    """
    batch_size = camera_matrix.size(0)
    device = camera_matrix.device
    
    # Create pixel coordinates
    i, j = torch.meshgrid(
        torch.linspace(0, image_width - 1, image_width, device=device),
        torch.linspace(0, image_height - 1, image_height, device=device),
        indexing='xy'
    )
    
    # Convert to normalized device coordinates [-1, 1]
    # Center the coordinates and normalize
    aspect_ratio = image_width / image_height
    fov_rad = fov_degrees * math.pi / 180.0
    focal_length = 1.0 / math.tan(fov_rad / 2.0)
    
    # Normalized coordinates
    x = (i - image_width / 2) / (image_width / 2) * aspect_ratio / focal_length
    y = -(j - image_height / 2) / (image_height / 2) / focal_length  # Flip y for image coordinates
    
    # Create direction vectors in camera space
    directions_camera = torch.stack([x, y, -torch.ones_like(x)], dim=-1)  # (H, W, 3)
    directions_camera = F.normalize(directions_camera, dim=-1)
    
    # Transform to world space using camera matrix
    # Extract rotation part (3x3)
    rotation_matrices = camera_matrix[:, :3, :3]  # (B, 3, 3)
    camera_positions = camera_matrix[:, :3, 3]    # (B, 3)
    
    # Expand directions for batch processing
    directions_camera_batch = directions_camera.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B, H, W, 3)
    
    # Apply rotation to get world-space directions
    # Reshape for batch matrix multiplication
    dirs_flat = directions_camera_batch.view(batch_size, -1, 3)  # (B, H*W, 3)
    dirs_world_flat = torch.bmm(dirs_flat, rotation_matrices.transpose(-1, -2))  # (B, H*W, 3)
    ray_directions = dirs_world_flat.view(batch_size, image_height, image_width, 3)
    
    # Ray origins are the camera positions, expanded for all pixels
    ray_origins = camera_positions.view(batch_size, 1, 1, 3).expand(-1, image_height, image_width, -1)
    
    return ray_origins, ray_directions


def sample_points_along_rays(
    ray_origins: Tensor,
    ray_directions: Tensor,
    near: float,
    far: float,
    num_samples: int,
    perturb: bool = True,
) -> Tuple[Tensor, Tensor]:
    """
    Sample points along rays for volumetric rendering.
    
    Args:
        ray_origins: Ray starting points (B, H, W, 3)
        ray_directions: Ray directions (B, H, W, 3)
        near: Near clipping distance
        far: Far clipping distance
        num_samples: Number of samples along each ray
        perturb: Whether to add random perturbation to sample positions
        
    Returns:
        Tuple of (sample_points, sample_distances)
        - sample_points: 3D sample points (B, H, W, num_samples, 3)
        - sample_distances: Distances along rays (B, H, W, num_samples)
    """
    batch_size, height, width = ray_origins.shape[:3]
    device = ray_origins.device
    
    # Create uniform sampling distances
    t_vals = torch.linspace(near, far, num_samples, device=device)
    t_vals = t_vals.expand(batch_size, height, width, num_samples)
    
    # Add perturbation for training
    if perturb:
        # Get intervals between samples
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], dim=-1)
        lower = torch.cat([t_vals[..., :1], mids], dim=-1)
        
        # Uniform samples in those intervals
        t_rand = torch.rand_like(t_vals)
        t_vals = lower + (upper - lower) * t_rand
    
    # Compute sample points: origin + t * direction
    sample_points = (
        ray_origins.unsqueeze(-2) + 
        t_vals.unsqueeze(-1) * ray_directions.unsqueeze(-2)
    )  # (B, H, W, num_samples, 3)
    
    return sample_points, t_vals


class CameraPoseRegressor(nn.Module):
    """
    Neural network to regress camera pose from images.
    Used by the discriminator for 3D consistency loss.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dims: Tuple[int, ...] = (256, 128),
        output_dim: int = 6,  # 3 for position, 3 for rotation (axis-angle)
    ):
        """
        Initialize camera pose regressor.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: Hidden layer dimensions
            output_dim: Output dimension (typically 6 for pose)
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize final layer with small weights
        nn.init.normal_(self.network[-1].weight, std=0.01)
        nn.init.zeros_(self.network[-1].bias)
    
    def forward(self, features: Tensor) -> Tensor:
        """
        Predict camera pose from image features.
        
        Args:
            features: Input features (B, input_dim)
            
        Returns:
            Predicted pose parameters (B, output_dim)
        """
        return self.network(features)


def camera_pose_loss(
    predicted_pose: Tensor,
    true_angles: Tensor,
    pose_weight: float = 1.0,
) -> Tensor:
    """
    Compute camera pose prediction loss.
    
    Args:
        predicted_pose: Predicted camera parameters (B, 6)
        true_angles: True camera angles [elevation, azimuth, radius] (B, 3)
        pose_weight: Weight for the pose loss
        
    Returns:
        Camera pose loss
    """
    # Extract predicted parameters
    pred_translation = predicted_pose[:, :3]  # (B, 3)
    pred_rotation = predicted_pose[:, 3:]     # (B, 3)
    
    # Convert true angles to translation (simplified version)
    # In practice, you might want a more sophisticated conversion
    true_elevation = true_angles[:, 0] * math.pi / 180.0
    true_azimuth = true_angles[:, 1] * math.pi / 180.0
    true_radius = true_angles[:, 2]
    
    true_x = true_radius * torch.cos(true_elevation) * torch.cos(true_azimuth)
    true_y = true_radius * torch.sin(true_elevation)
    true_z = true_radius * torch.cos(true_elevation) * torch.sin(true_azimuth)
    true_translation = torch.stack([true_x, true_y, true_z], dim=1)
    
    # Compute losses
    translation_loss = F.mse_loss(pred_translation, true_translation)
    rotation_loss = F.mse_loss(pred_rotation, torch.zeros_like(pred_rotation))  # Simplified
    
    total_loss = pose_weight * (translation_loss + rotation_loss)
    
    return total_loss