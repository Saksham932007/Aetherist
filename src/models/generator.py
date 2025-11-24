"""
Generator architecture for Aetherist Hybrid 3D-ViT-GAN.
Combines Vision Transformer backbone with Neural Radiance Field rendering.
"""

import math
from typing import Optional, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import einops
from einops import rearrange, repeat

from .mapping import MappingNetwork
from .vit import VisionTransformer
from .layers import AdaptiveLayerNorm, FusedMLP


class TriPlaneHead(nn.Module):
    """
    Output head that projects transformer features into tri-plane representation.
    
    Tri-planes decompose 3D features into three orthogonal 2D planes:
    - XY plane: Features along the XY spatial dimensions
    - XZ plane: Features along the XZ spatial dimensions  
    - YZ plane: Features along the YZ spatial dimensions
    
    This representation allows efficient neural rendering with reduced memory.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        triplane_channels: int = 32,
        triplane_resolution: int = 256,
        num_layers: int = 3,
        hidden_dim: Optional[int] = None,
        activation: str = "leaky_relu",
        use_spectral_norm: bool = False,
    ):
        """
        Initialize tri-plane head.
        
        Args:
            input_dim: Input feature dimensionality from ViT
            triplane_channels: Number of channels per tri-plane
            triplane_resolution: Resolution of each tri-plane (H, W)
            num_layers: Number of projection layers
            hidden_dim: Hidden layer dimensionality
            activation: Activation function
            use_spectral_norm: Whether to use spectral normalization
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.triplane_channels = triplane_channels
        self.triplane_resolution = triplane_resolution
        self.num_layers = num_layers
        
        hidden_dim = hidden_dim or input_dim
        
        # Calculate output size for each plane
        plane_size = triplane_resolution * triplane_resolution * triplane_channels
        total_output_size = 3 * plane_size  # 3 planes: XY, XZ, YZ
        
        # Build projection network
        layers = []
        
        # First layer
        linear = nn.Linear(input_dim, hidden_dim)
        if use_spectral_norm:
            linear = nn.utils.spectral_norm(linear)
        layers.extend([
            linear,
            self._get_activation(activation),
        ])
        
        # Hidden layers
        for _ in range(num_layers - 2):
            linear = nn.Linear(hidden_dim, hidden_dim)
            if use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.extend([
                linear,
                self._get_activation(activation),
            ])
        
        # Final layer to tri-plane features
        final_linear = nn.Linear(hidden_dim, total_output_size)
        if use_spectral_norm:
            final_linear = nn.utils.spectral_norm(final_linear)
        layers.append(final_linear)
        
        self.projection = nn.Sequential(*layers)
        
        # Initialize final layer with small weights
        nn.init.normal_(self.projection[-1].weight, std=0.01)
        nn.init.zeros_(self.projection[-1].bias)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation == "leaky_relu":
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, features: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass to generate tri-plane representation.
        
        Args:
            features: Input features (B, N, input_dim) from ViT
            
        Returns:
            Dictionary containing:
            - 'xy_plane': XY plane features (B, C, H, W)
            - 'xz_plane': XZ plane features (B, C, H, W)
            - 'yz_plane': YZ plane features (B, C, H, W)
        """
        batch_size = features.size(0)
        
        # Global average pooling to get single feature vector per batch
        # Alternative: use cls token or learned aggregation
        pooled_features = features.mean(dim=1)  # (B, input_dim)
        
        # Project to tri-plane features
        triplane_flat = self.projection(pooled_features)  # (B, 3 * H * W * C)
        
        # Reshape to individual planes
        plane_size = self.triplane_resolution * self.triplane_resolution * self.triplane_channels
        
        # Split into three planes
        xy_flat = triplane_flat[:, :plane_size]
        xz_flat = triplane_flat[:, plane_size:2*plane_size]
        yz_flat = triplane_flat[:, 2*plane_size:]
        
        # Reshape to 2D planes
        xy_plane = xy_flat.view(batch_size, self.triplane_channels, 
                               self.triplane_resolution, self.triplane_resolution)
        xz_plane = xz_flat.view(batch_size, self.triplane_channels,
                               self.triplane_resolution, self.triplane_resolution)
        yz_plane = yz_flat.view(batch_size, self.triplane_channels,
                               self.triplane_resolution, self.triplane_resolution)
        
        return {
            'xy_plane': xy_plane,
            'xz_plane': xz_plane,
            'yz_plane': yz_plane,
        }


class NeuralRenderer(nn.Module):
    """
    Neural renderer for volumetric ray-marching.
    
    Implements the core NeRF rendering equation:
    C(r) = ∑_{i=1}^{N} T_i * (1 - exp(-σ_i * δ_i)) * c_i
    
    Where:
    - T_i = exp(-∑_{j=1}^{i-1} σ_j * δ_j) is the transmittance
    - σ_i is the volume density at point i
    - c_i is the color at point i
    - δ_i is the distance between adjacent samples
    """
    
    def __init__(
        self,
        triplane_channels: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 4,
        output_channels: int = 4,  # RGB + density
        activation: str = "relu",
        use_viewdir: bool = True,
        viewdir_dim: int = 3,
    ):
        """
        Initialize neural renderer.
        
        Args:
            triplane_channels: Number of channels in tri-plane features
            hidden_dim: Hidden layer dimensionality
            num_layers: Number of MLP layers
            output_channels: Output channels (3 for RGB + 1 for density)
            activation: Activation function
            use_viewdir: Whether to condition on view direction
            viewdir_dim: View direction dimensionality
        """
        super().__init__()
        
        self.triplane_channels = triplane_channels
        self.use_viewdir = use_viewdir
        self.output_channels = output_channels
        
        # Input dimensionality: features from 3 planes
        input_dim = 3 * triplane_channels
        if use_viewdir:
            input_dim += viewdir_dim
        
        # MLP for density and features
        density_layers = []
        prev_dim = input_dim
        
        for _ in range(num_layers - 1):
            density_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                self._get_activation(activation),
            ])
            prev_dim = hidden_dim
        
        # Density output
        density_layers.append(nn.Linear(prev_dim, 1))  # Single density value
        self.density_net = nn.Sequential(*density_layers)
        
        # Color MLP (conditioned on density features + view direction)
        if use_viewdir:
            color_input_dim = hidden_dim + viewdir_dim
        else:
            color_input_dim = hidden_dim
        
        self.color_net = nn.Sequential(
            nn.Linear(color_input_dim, hidden_dim // 2),
            self._get_activation(activation),
            nn.Linear(hidden_dim // 2, 3),  # RGB color
        )
        
        # Feature extraction layer (before final density prediction)
        self.feature_layer = nn.Linear(prev_dim, hidden_dim)
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def sample_from_triplanes(
        self,
        points: Tensor,
        triplanes: Dict[str, Tensor],
    ) -> Tensor:
        """
        Sample features from tri-planes at given 3D points.
        
        Args:
            points: 3D points (B, N, 3) in normalized coordinates [-1, 1]
            triplanes: Dictionary of tri-plane features
            
        Returns:
            Sampled features (B, N, 3 * triplane_channels)
        """
        batch_size, num_points = points.shape[:2]
        
        # Normalize points to [-1, 1] for grid sampling
        x, y, z = points.unbind(-1)  # Each: (B, N)
        
        # Sample from XY plane (using x, y coordinates)
        xy_coords = torch.stack([x, y], dim=-1).unsqueeze(-2)  # (B, N, 1, 2)
        xy_features = F.grid_sample(
            triplanes['xy_plane'], xy_coords, 
            mode='bilinear', padding_mode='border', align_corners=False
        )  # (B, C, N, 1)
        xy_features = xy_features.squeeze(-1).transpose(1, 2)  # (B, N, C)
        
        # Sample from XZ plane (using x, z coordinates)
        xz_coords = torch.stack([x, z], dim=-1).unsqueeze(-2)  # (B, N, 1, 2)
        xz_features = F.grid_sample(
            triplanes['xz_plane'], xz_coords,
            mode='bilinear', padding_mode='border', align_corners=False
        )  # (B, C, N, 1)
        xz_features = xz_features.squeeze(-1).transpose(1, 2)  # (B, N, C)
        
        # Sample from YZ plane (using y, z coordinates)
        yz_coords = torch.stack([y, z], dim=-1).unsqueeze(-2)  # (B, N, 1, 2)
        yz_features = F.grid_sample(
            triplanes['yz_plane'], yz_coords,
            mode='bilinear', padding_mode='border', align_corners=False
        )  # (B, C, N, 1)
        yz_features = yz_features.squeeze(-1).transpose(1, 2)  # (B, N, C)
        
        # Concatenate features from all three planes
        combined_features = torch.cat([xy_features, xz_features, yz_features], dim=-1)
        
        return combined_features  # (B, N, 3 * triplane_channels)
    
    def forward(
        self,
        points: Tensor,
        viewdirs: Optional[Tensor],
        triplanes: Dict[str, Tensor],
    ) -> Dict[str, Tensor]:
        """
        Render colors and densities for given points and view directions.
        
        Args:
            points: 3D points (B, N, 3)
            viewdirs: View directions (B, N, 3) if use_viewdir=True
            triplanes: Tri-plane features
            
        Returns:
            Dictionary containing:
            - 'rgb': RGB colors (B, N, 3)
            - 'density': Volume densities (B, N, 1)
        """
        # Sample features from tri-planes
        triplane_features = self.sample_from_triplanes(points, triplanes)
        
        # Prepare input for density network
        density_input = triplane_features
        if self.use_viewdir and viewdirs is not None:
            density_input = torch.cat([density_input, viewdirs], dim=-1)
        
        # Forward through density network (excluding final layer)
        x = density_input
        for layer in self.density_net[:-1]:
            x = layer(x)
        
        # Extract features before density prediction
        features = self.feature_layer(x)
        
        # Predict density
        density = self.density_net[-1](x)
        density = F.softplus(density - 1)  # Shift softplus for better initialization
        
        # Predict color
        if self.use_viewdir and viewdirs is not None:
            color_input = torch.cat([features, viewdirs], dim=-1)
        else:
            color_input = features
        
        rgb = self.color_net(color_input)
        rgb = torch.sigmoid(rgb)  # Ensure RGB values in [0, 1]
        
        return {
            'rgb': rgb,
            'density': density,
        }


class SuperResolutionUpsampler(nn.Module):
    """
    CNN-based super-resolution upsampler.
    
    Takes low-resolution neural renders and upsamples them to target resolution
    while adding fine details and reducing artifacts.
    """
    
    def __init__(
        self,
        input_resolution: int = 64,
        output_resolution: int = 256,
        input_channels: int = 3,
        base_channels: int = 64,
        num_blocks: int = 4,
        use_spectral_norm: bool = True,
        activation: str = "leaky_relu",
    ):
        """
        Initialize super-resolution upsampler.
        
        Args:
            input_resolution: Input image resolution
            output_resolution: Target output resolution  
            input_channels: Number of input channels (usually 3 for RGB)
            base_channels: Base number of channels
            num_blocks: Number of residual blocks
            use_spectral_norm: Whether to use spectral normalization
            activation: Activation function
        """
        super().__init__()
        
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.scale_factor = output_resolution // input_resolution
        
        assert output_resolution % input_resolution == 0, \
            f"Output resolution {output_resolution} must be multiple of input resolution {input_resolution}"
        
        # Calculate number of upsampling stages
        self.num_upsample_stages = int(math.log2(self.scale_factor))
        
        # Input convolution
        self.input_conv = self._make_conv_layer(
            input_channels, base_channels, 7, 1, 3, use_spectral_norm
        )
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.residual_blocks.append(
                ResidualBlock(base_channels, base_channels, use_spectral_norm, activation)
            )
        
        # Upsampling layers
        self.upsample_layers = nn.ModuleList()
        current_channels = base_channels
        
        for i in range(self.num_upsample_stages):
            # Halve channels with each upsampling stage
            out_channels = max(base_channels // (2 ** (i + 1)), base_channels // 8)
            
            self.upsample_layers.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                self._make_conv_layer(current_channels, out_channels, 3, 1, 1, use_spectral_norm),
                self._get_activation(activation),
            ))
            current_channels = out_channels
        
        # Output convolution
        self.output_conv = nn.Sequential(
            self._make_conv_layer(current_channels, input_channels, 7, 1, 3, False),
            nn.Tanh(),
        )
    
    def _make_conv_layer(self, in_ch, out_ch, kernel_size, stride, padding, use_spectral_norm):
        """Create convolution layer with optional spectral normalization."""
        conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=True)
        if use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)
        return conv
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function."""
        if activation == "leaky_relu":
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of super-resolution upsampler.
        
        Args:
            x: Input low-resolution images (B, 3, H, W)
            
        Returns:
            Upsampled high-resolution images (B, 3, H*scale, W*scale)
        """
        # Input convolution
        x = self.input_conv(x)
        x = self._get_activation("leaky_relu")(x)
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Upsampling stages
        for upsample_layer in self.upsample_layers:
            x = upsample_layer(x)
        
        # Output convolution
        x = self.output_conv(x)
        
        return x


class ResidualBlock(nn.Module):
    """Residual block for super-resolution network."""
    
    def __init__(self, channels: int, out_channels: int, use_spectral_norm: bool, activation: str):
        super().__init__()
        
        conv1 = nn.Conv2d(channels, out_channels, 3, 1, 1, bias=True)
        conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True)
        
        if use_spectral_norm:
            conv1 = nn.utils.spectral_norm(conv1)
            conv2 = nn.utils.spectral_norm(conv2)
        
        if activation == "leaky_relu":
            act = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "relu":
            act = nn.ReLU(inplace=True)
        else:
            act = nn.GELU()
        
        self.block = nn.Sequential(
            conv1,
            act,
            conv2,
        )
        
        # Skip connection
        if channels != out_channels:
            skip_conv = nn.Conv2d(channels, out_channels, 1, 1, 0, bias=False)
            if use_spectral_norm:
                skip_conv = nn.utils.spectral_norm(skip_conv)
            self.skip = skip_conv
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        return self.skip(x) + self.block(x)


class AetheristGenerator(nn.Module):
    """
    Main generator architecture for Aetherist.
    
    Architecture Overview:
    1. Mapping Network: z → w (latent code transformation)
    2. Vision Transformer: Style-conditioned feature generation  
    3. Tri-Plane Head: Project features to 3D tri-plane representation
    4. Neural Renderer: Volumetric ray-marching for 2D image generation
    5. Super-Resolution: CNN upsampler for final high-resolution output
    """
    
    def __init__(
        self,
        # Latent space configuration
        latent_dim: int = 512,
        style_dim: int = 512,
        
        # Mapping network configuration
        mapping_layers: int = 8,
        
        # Vision Transformer configuration
        vit_layers: int = 12,
        vit_heads: int = 16,
        vit_dim: int = 1024,
        vit_mlp_ratio: float = 4.0,
        
        # Tri-plane configuration
        triplane_resolution: int = 256,
        triplane_channels: int = 32,
        
        # Renderer configuration
        renderer_hidden: int = 128,
        renderer_layers: int = 4,
        
        # Text conditioning
        use_text_conditioning: bool = False,
        text_dim: int = 512,
        
        # Training configuration
        use_siren: bool = True,
        siren_omega: float = 30.0,
    ):
        """
        Initialize Aetherist generator.
        
        Args:
            latent_dim: Input latent code dimensionality
            style_dim: Style vector dimensionality
            mapping_layers: Number of mapping network layers
            vit_layers: Number of ViT layers
            vit_heads: Number of ViT attention heads
            vit_dim: ViT feature dimensionality
            vit_mlp_ratio: ViT MLP expansion ratio
            triplane_resolution: Resolution of tri-plane features
            triplane_channels: Number of channels per tri-plane
            renderer_hidden: Renderer hidden dimensionality
            renderer_layers: Number of renderer layers
            use_text_conditioning: Whether to use text conditioning
            text_dim: Text embedding dimensionality
            use_siren: Whether to use SIREN activations
            siren_omega: SIREN frequency parameter
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.style_dim = style_dim
        self.triplane_resolution = triplane_resolution
        self.triplane_channels = triplane_channels
        self.use_text_conditioning = use_text_conditioning
        
        # Mapping network
        self.mapping_network = MappingNetwork(
            latent_dim=latent_dim,
            hidden_dim=style_dim,
            num_layers=mapping_layers,
            use_text_conditioning=use_text_conditioning,
            text_embedding_dim=text_dim,
        )
        
        # Vision Transformer backbone
        # Note: For generation, we don't use patch embeddings from images
        # Instead, we'll use learnable feature tokens
        self.num_tokens = (triplane_resolution // 16) ** 2  # Equivalent to 16x16 patches
        self.feature_tokens = nn.Parameter(torch.randn(1, self.num_tokens, vit_dim) * 0.02)
        
        self.vit_backbone = VisionTransformer(
            img_size=triplane_resolution,
            patch_size=16,
            in_channels=3,  # Won't be used since we use learnable tokens
            embed_dim=vit_dim,
            num_layers=vit_layers,
            num_heads=vit_heads,
            mlp_ratio=vit_mlp_ratio,
            style_dim=style_dim,
            use_cross_attention=use_text_conditioning,
            text_dim=text_dim,
            use_siren=use_siren,
            class_token=False,
        )
        
        # Tri-plane head
        self.triplane_head = TriPlaneHead(
            input_dim=vit_dim,
            triplane_channels=triplane_channels,
            triplane_resolution=triplane_resolution,
        )
        
        # Neural renderer
        self.neural_renderer = NeuralRenderer(
            triplane_channels=triplane_channels,
            hidden_dim=renderer_hidden,
            num_layers=renderer_layers,
        )
        
        # Super-resolution upsampler
        self.sr_upsampler = SuperResolutionUpsampler(
            input_resolution=64,  # Low-res neural render
            output_resolution=256,  # High-res final output
            input_channels=3,
        )
        
        # Initialize feature tokens
        nn.init.trunc_normal_(self.feature_tokens, std=0.02)
    
    def forward(
        self,
        z: Tensor,
        camera_matrix: Tensor,
        text_embeddings: Optional[Tensor] = None,
        truncation_psi: float = 1.0,
        return_triplanes: bool = False,
        render_resolution: int = 64,
        final_resolution: int = 256,
        **render_kwargs,
    ) -> Dict[str, Tensor]:
        """
        Complete forward pass of the generator.
        
        Args:
            z: Input latent codes (B, latent_dim)
            camera_matrix: Camera transformation matrices (B, 4, 4)
            text_embeddings: Optional text embeddings (B, text_dim)
            truncation_psi: Truncation parameter
            return_triplanes: Whether to return tri-plane features
            render_resolution: Resolution for neural rendering
            final_resolution: Final output resolution after super-resolution
            **render_kwargs: Additional rendering arguments
            
        Returns:
            Dictionary containing generated results including final images
        """
        batch_size = z.size(0)
        
        # Map latent codes to style vectors
        w = self.mapping_network(z, text_embeddings, truncation_psi)
        
        # Expand feature tokens for batch
        feature_tokens = self.feature_tokens.expand(batch_size, -1, -1)
        
        # Process feature tokens through ViT layers directly (skip patch embedding)
        # Since we're using learnable tokens, we bypass patch embedding
        x = feature_tokens
        B = x.shape[0]
        
        # Add positional encoding
        x = self.vit_backbone.pos_encoding(x)
        
        # Apply transformer blocks with style conditioning
        for block in self.vit_backbone.blocks:
            x = block(x, style=w, text_embeddings=text_embeddings)
        
        # Apply final layer norm with style
        vit_features = self.vit_backbone.norm(x, style=w)
        
        # Generate tri-plane features
        triplanes = self.triplane_head(vit_features)
        
        # Neural rendering at low resolution
        low_res_image = self.render_image(
            triplanes=triplanes,
            camera_matrix=camera_matrix,
            image_height=render_resolution,
            image_width=render_resolution,
            **render_kwargs
        )
        
        # Super-resolution upsampling
        # Convert to [-1, 1] range for upsampler
        low_res_normalized = low_res_image * 2.0 - 1.0
        high_res_image = self.sr_upsampler(low_res_normalized)
        # Convert back to [0, 1] range
        high_res_image = (high_res_image + 1.0) / 2.0
        high_res_image = torch.clamp(high_res_image, 0.0, 1.0)
        
        # Prepare output dictionary
        output = {
            'w': w,
            'vit_features': vit_features,
            'low_res_image': low_res_image,
            'high_res_image': high_res_image,
            'final_image': high_res_image,  # Alias for compatibility
        }
        
        if return_triplanes:
            output['triplanes'] = triplanes
        
        return output
    
    def render_image(
        self,
        triplanes: Dict[str, Tensor],
        camera_matrix: Tensor,
        image_height: int = 256,
        image_width: int = 256,
        num_samples: int = 64,
        num_importance_samples: int = 128,
        near_plane: float = 0.5,
        far_plane: float = 2.5,
        white_background: bool = True,
        **render_kwargs,
    ) -> Tensor:
        """
        Render 2D image from tri-plane features using volumetric ray-marching.
        
        Implements the core NeRF rendering equation:
        C(r) = ∑_{i=1}^{N} T_i * (1 - exp(-σ_i * δ_i)) * c_i
        
        Args:
            triplanes: Tri-plane features
            camera_matrix: Camera transformation matrix (B, 4, 4)
            image_height: Output image height
            image_width: Output image width
            num_samples: Number of coarse samples per ray
            num_importance_samples: Number of fine samples per ray
            near_plane: Near clipping distance
            far_plane: Far clipping distance
            white_background: Whether to use white background
            **render_kwargs: Additional rendering arguments
            
        Returns:
            Rendered images (B, 3, H, W)
        """
        from ..utils.camera import generate_rays, sample_points_along_rays
        
        batch_size = list(triplanes.values())[0].size(0)
        device = list(triplanes.values())[0].device
        
        # Generate rays from camera
        ray_origins, ray_directions = generate_rays(
            camera_matrix, image_height, image_width,
            fov_degrees=50.0, near=near_plane, far=far_plane
        )
        
        # Sample points along rays (coarse sampling)
        sample_points, sample_distances = sample_points_along_rays(
            ray_origins, ray_directions, near_plane, far_plane,
            num_samples, perturb=self.training
        )
        
        # Flatten for neural network processing
        B, H, W, N, _ = sample_points.shape
        points_flat = sample_points.view(B, H * W * N, 3)  # (B, HWN, 3)
        
        # Normalize points to [-1, 1] for tri-plane sampling
        points_normalized = points_flat / far_plane  # Simple normalization
        points_normalized = torch.clamp(points_normalized, -1, 1)
        
        # Prepare view directions
        viewdirs_flat = ray_directions.unsqueeze(-2).expand(-1, -1, -1, N, -1)
        viewdirs_flat = viewdirs_flat.contiguous().view(B, H * W * N, 3)
        viewdirs_flat = F.normalize(viewdirs_flat, dim=-1)
        
        # Render colors and densities
        render_output = self.neural_renderer(
            points_normalized, viewdirs_flat, triplanes
        )
        
        rgb = render_output['rgb'].view(B, H, W, N, 3)  # (B, H, W, N, 3)
        density = render_output['density'].view(B, H, W, N, 1)  # (B, H, W, N, 1)
        
        # Compute distances between samples
        dists = sample_distances[..., 1:] - sample_distances[..., :-1]  # (B, H, W, N-1)
        dists = torch.cat([
            dists,
            torch.full_like(dists[..., :1], 1e10)  # Infinite distance for last sample
        ], dim=-1)
        
        # Volume rendering equation
        # α = 1 - exp(-σ * δ)
        alpha = 1.0 - torch.exp(-density.squeeze(-1) * dists)  # (B, H, W, N)
        
        # Transmittance T = exp(-∑σδ) = ∏(1-α)
        # Use cumprod for efficient computation
        transmittance = torch.cumprod(torch.cat([
            torch.ones_like(alpha[..., :1]),
            1.0 - alpha[..., :-1]
        ], dim=-1), dim=-1)
        
        # Weights for color composition
        weights = transmittance * alpha  # (B, H, W, N)
        
        # Composite colors
        rendered_rgb = torch.sum(weights.unsqueeze(-1) * rgb, dim=-2)  # (B, H, W, 3)
        
        # Add white background if specified
        if white_background:
            acc_transmittance = torch.sum(weights, dim=-1, keepdim=True)  # (B, H, W, 1)
            rendered_rgb = rendered_rgb + (1.0 - acc_transmittance)
        
        # Convert to image format (B, 3, H, W)
        rendered_image = rendered_rgb.permute(0, 3, 1, 2)
        
        return torch.clamp(rendered_image, 0.0, 1.0)


def test_generator_components():
    """Test function for generator components."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 2
    latent_dim = 512
    triplane_channels = 32
    triplane_resolution = 64
    
    print("Testing Tri-Plane Head...")
    
    # Test tri-plane head
    triplane_head = TriPlaneHead(
        input_dim=256,
        triplane_channels=triplane_channels,
        triplane_resolution=triplane_resolution,
    ).to(device)
    
    seq_len = 16
    features = torch.randn(batch_size, seq_len, 256, device=device)
    triplanes = triplane_head(features)
    
    print(f"Tri-plane head input: {features.shape}")
    for name, plane in triplanes.items():
        print(f"{name}: {plane.shape}")
        assert plane.shape == (batch_size, triplane_channels, triplane_resolution, triplane_resolution)
    
    print("Testing Neural Renderer...")
    
    # Test neural renderer
    renderer = NeuralRenderer(triplane_channels=triplane_channels).to(device)
    
    num_points = 1024
    points = torch.randn(batch_size, num_points, 3, device=device) * 0.5  # Normalized coordinates
    viewdirs = torch.randn(batch_size, num_points, 3, device=device)
    viewdirs = F.normalize(viewdirs, dim=-1)
    
    render_output = renderer(points, viewdirs, triplanes)
    
    print(f"Renderer input points: {points.shape}")
    print(f"RGB output: {render_output['rgb'].shape}")
    print(f"Density output: {render_output['density'].shape}")
    assert render_output['rgb'].shape == (batch_size, num_points, 3)
    assert render_output['density'].shape == (batch_size, num_points, 1)
    
    print("Testing Super-Resolution Upsampler...")
    
    # Test super-resolution upsampler
    upsampler = SuperResolutionUpsampler(
        input_resolution=64,
        output_resolution=256,
        input_channels=3,
    ).to(device)
    
    low_res_img = torch.randn(batch_size, 3, 64, 64, device=device)
    high_res_img = upsampler(low_res_img)
    
    print(f"Upsampler input: {low_res_img.shape}")
    print(f"Upsampler output: {high_res_img.shape}")
    assert high_res_img.shape == (batch_size, 3, 256, 256)
    
    print("Testing Full Generator Pipeline...")
    
    # Test full generator
    generator = AetheristGenerator(
        vit_dim=256,
        vit_layers=4,
        triplane_resolution=triplane_resolution,
        triplane_channels=triplane_channels,
    ).to(device)
    
    z = torch.randn(batch_size, latent_dim, device=device)
    
    # Create dummy camera matrices
    from ..utils.camera import sample_camera_poses, look_at_matrix, perspective_projection_matrix
    
    # Sample camera poses
    eye_positions, view_matrices, camera_angles = sample_camera_poses(
        batch_size, device=device
    )
    
    # Create projection matrices
    proj_matrices = perspective_projection_matrix(
        fov_degrees=torch.tensor([50.0] * batch_size),
        aspect_ratio=torch.tensor([1.0] * batch_size),
    ).to(device)
    
    # Combine view and projection matrices
    camera_matrices = torch.bmm(proj_matrices, view_matrices)
    
    output = generator(z, camera_matrices, return_triplanes=True)
    
    print(f"Generator input: {z.shape}")
    print(f"Style vectors: {output['w'].shape}")
    print(f"ViT features: {output['vit_features'].shape}")
    print(f"Low-res image: {output['low_res_image'].shape}")
    print(f"High-res image: {output['high_res_image'].shape}")
    
    # Verify output shapes
    assert output['low_res_image'].shape == (batch_size, 3, 64, 64)
    assert output['high_res_image'].shape == (batch_size, 3, 256, 256)
    
    triplanes = output['triplanes']
    for name, plane in triplanes.items():
        print(f"{name}: {plane.shape}")
    
    print("All generator component tests passed!")
    print("✅ Complete Aetherist Generator pipeline working!")


if __name__ == "__main__":
    test_generator_components()