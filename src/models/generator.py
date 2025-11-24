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


class AetheristGenerator(nn.Module):
    """
    Main generator architecture for Aetherist.
    
    Architecture Overview:
    1. Mapping Network: z → w (latent code transformation)
    2. Vision Transformer: Style-conditioned feature generation  
    3. Tri-Plane Head: Project features to 3D tri-plane representation
    4. Neural Renderer: Volumetric ray-marching for 2D image generation
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
        
        # Initialize feature tokens
        nn.init.trunc_normal_(self.feature_tokens, std=0.02)
    
    def forward(
        self,
        z: Tensor,
        camera_angles: Tensor,
        text_embeddings: Optional[Tensor] = None,
        truncation_psi: float = 1.0,
        return_triplanes: bool = False,
        **render_kwargs,
    ) -> Dict[str, Tensor]:
        """
        Forward pass of the generator.
        
        Args:
            z: Input latent codes (B, latent_dim)
            camera_angles: Camera angles [elevation, azimuth, radius] (B, 3)
            text_embeddings: Optional text embeddings (B, text_dim)
            truncation_psi: Truncation parameter
            return_triplanes: Whether to return tri-plane features
            **render_kwargs: Additional rendering arguments
            
        Returns:
            Dictionary containing generated results
        """
        batch_size = z.size(0)
        
        # Map latent codes to style vectors
        w = self.mapping_network(z, text_embeddings, truncation_psi)
        
        # Expand feature tokens for batch
        feature_tokens = self.feature_tokens.expand(batch_size, -1, -1)
        
        # Process through ViT with style conditioning
        vit_features = self.vit_backbone(
            feature_tokens,  # Use learnable tokens instead of image patches
            style=w,
            text_embeddings=text_embeddings,
        )
        
        # Generate tri-plane features
        triplanes = self.triplane_head(vit_features)
        
        # Prepare output dictionary
        output = {
            'w': w,
            'vit_features': vit_features,
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
        **render_kwargs,
    ) -> Tensor:
        """
        Render 2D image from tri-plane features.
        
        This method will be completed in the neural renderer implementation.
        For now, it returns a placeholder.
        
        Args:
            triplanes: Tri-plane features
            camera_matrix: Camera transformation matrix (B, 4, 4)
            image_height: Output image height
            image_width: Output image width
            **render_kwargs: Additional rendering arguments
            
        Returns:
            Rendered images (B, 3, H, W)
        """
        batch_size = list(triplanes.values())[0].size(0)
        
        # Placeholder: return random images
        # This will be replaced with actual ray-marching implementation
        return torch.randn(batch_size, 3, image_height, image_width, 
                          device=list(triplanes.values())[0].device)


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
    
    print("Testing Full Generator...")
    
    # Test full generator
    generator = AetheristGenerator(
        vit_dim=256,
        vit_layers=4,
        triplane_resolution=triplane_resolution,
        triplane_channels=triplane_channels,
    ).to(device)
    
    z = torch.randn(batch_size, latent_dim, device=device)
    camera_angles = torch.randn(batch_size, 3, device=device)
    
    output = generator(z, camera_angles, return_triplanes=True)
    
    print(f"Generator input: {z.shape}")
    print(f"Style vectors: {output['w'].shape}")
    print(f"ViT features: {output['vit_features'].shape}")
    
    triplanes = output['triplanes']
    for name, plane in triplanes.items():
        print(f"{name}: {plane.shape}")
    
    print("All generator component tests passed!")


if __name__ == "__main__":
    test_generator_components()