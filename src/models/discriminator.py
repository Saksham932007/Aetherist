"""
Discriminator architecture for Aetherist.

This module implements a dual-branch discriminator that evaluates both image quality 
and 3D consistency. The discriminator uses ViT-based architecture to provide 
detailed feedback for training the Aetherist generator.

Architecture Overview:
1. Image Quality Branch: Evaluates photorealism and visual quality
2. 3D Consistency Branch: Assesses 3D geometric consistency across viewpoints
3. Multi-scale Analysis: Processes images at different resolutions
4. Spectral Normalization: Ensures training stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Union
import math

from .layers import ModulatedLinear, AdaptiveLayerNorm
from .attention import MultiHeadSelfAttention, WindowAttention


class SpectralNorm(nn.Module):
    """
    Spectral normalization wrapper for stable discriminator training.
    
    Spectral normalization constrains the Lipschitz constant of the network,
    which is crucial for stable GAN training and prevents mode collapse.
    """
    
    def __init__(self, module: nn.Module, name: str = 'weight', power_iterations: int = 1):
        """
        Initialize spectral normalization.
        
        Args:
            module: Module to apply spectral normalization to
            name: Name of weight parameter to normalize
            power_iterations: Number of power iterations for spectral norm computation
        """
        super().__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        
        # Initialize spectral norm parameters
        if not self._made_params():
            self._make_params()
    
    def _update_u_v(self):
        """Update u and v vectors for spectral normalization."""
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")
        
        # Reshape weight matrix for spectral norm computation
        height = w.data.shape[0]
        w_mat = w.view(height, -1)  # Flatten to 2D matrix
        
        # Power iteration
        for _ in range(self.power_iterations):
            v = F.normalize(torch.mv(w_mat.t(), u), dim=0, eps=1e-12)
            u = F.normalize(torch.mv(w_mat, v), dim=0, eps=1e-12)
        
        # Compute spectral norm
        sigma = torch.dot(u, torch.mv(w_mat, v))
        setattr(self.module, self.name, w / sigma)
    
    def _made_params(self):
        """Check if spectral norm parameters exist."""
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False
    
    def _make_params(self):
        """Initialize spectral norm parameters."""
        w = getattr(self.module, self.name)
        
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]
        
        u = F.normalize(w.new_empty(height).normal_(0, 1), dim=0, eps=1e-12)
        v = F.normalize(w.new_empty(width).normal_(0, 1), dim=0, eps=1e-12)
        u = u.clone().detach().requires_grad_(False)
        v = v.clone().detach().requires_grad_(False)
        w_bar = w.clone().detach().requires_grad_(False)
        
        del self.module._parameters[self.name]
        self.module.register_parameter(self.name + "_u", nn.Parameter(u))
        self.module.register_parameter(self.name + "_v", nn.Parameter(v))
        self.module.register_parameter(self.name + "_bar", nn.Parameter(w_bar))
    
    def forward(self, *args, **kwargs):
        """Forward pass with spectral normalization."""
        self._update_u_v()
        return self.module(*args, **kwargs)


class DiscriminatorBlock(nn.Module):
    """
    Basic building block for the discriminator.
    
    Combines convolution, normalization, and activation with residual connections.
    Uses spectral normalization for training stability.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_spectral_norm: bool = True,
        activation: str = "leaky_relu",
        use_attention: bool = False,
        attention_heads: int = 8,
    ):
        """
        Initialize discriminator block.
        
        Args:
            in_channels: Input channels
            out_channels: Output channels
            kernel_size: Convolution kernel size
            stride: Convolution stride
            padding: Convolution padding
            use_spectral_norm: Whether to use spectral normalization
            activation: Activation function name
            use_attention: Whether to add self-attention
            attention_heads: Number of attention heads
        """
        super().__init__()
        
        self.use_attention = use_attention
        
        # Main convolution
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        if use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)
        self.conv = conv
        
        # Normalization (using instance norm for discriminator)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        
        # Activation
        if activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Optional self-attention
        if use_attention:
            self.attention = MultiHeadSelfAttention(
                dim=out_channels,
                num_heads=attention_heads,
                qkv_bias=True,
                attn_drop=0.1,
                proj_drop=0.1,
            )
        
        # Residual connection (if dimensions match)
        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            residual_conv = nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False)
            if use_spectral_norm:
                residual_conv = nn.utils.spectral_norm(residual_conv)
            self.residual = residual_conv
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Output tensor (B, C', H', W')
        """
        residual = self.residual(x)
        
        # Main path
        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        
        # Optional attention
        if self.use_attention:
            B, C, H, W = out.shape
            # Reshape for attention: (B, C, H, W) -> (B, H*W, C)
            out_reshaped = out.view(B, C, H * W).transpose(1, 2)
            out_reshaped = self.attention(out_reshaped)
            # Reshape back: (B, H*W, C) -> (B, C, H, W)
            out = out_reshaped.transpose(1, 2).view(B, C, H, W)
        
        # Residual connection
        return out + residual


class ImageQualityBranch(nn.Module):
    """
    Image Quality Branch of the discriminator.
    
    This branch evaluates the photorealism and visual quality of generated images.
    Uses a CNN backbone with self-attention for detailed analysis.
    """
    
    def __init__(
        self,
        input_size: int = 256,
        input_channels: int = 3,
        base_channels: int = 64,
        max_channels: int = 512,
        num_layers: int = 6,
        use_attention_layers: Optional[List[int]] = None,
    ):
        """
        Initialize image quality branch.
        
        Args:
            input_size: Input image size
            input_channels: Number of input channels
            base_channels: Base number of channels
            max_channels: Maximum number of channels
            num_layers: Number of convolutional layers
            use_attention_layers: Layers to add self-attention (None for [3, 4])
        """
        super().__init__()
        
        if use_attention_layers is None:
            use_attention_layers = [3, 4]  # Add attention in middle layers
        
        self.input_size = input_size
        self.num_layers = num_layers
        
        # Calculate channel progression
        channels = [input_channels]
        for i in range(num_layers):
            next_channels = min(base_channels * (2 ** i), max_channels)
            channels.append(next_channels)
        
        # Build layers
        layers = []
        for i in range(num_layers):
            stride = 2 if i < num_layers - 2 else 1  # Don't downsample in last 2 layers
            use_attention = i in use_attention_layers
            
            layer = DiscriminatorBlock(
                in_channels=channels[i],
                out_channels=channels[i + 1],
                stride=stride,
                use_attention=use_attention,
                attention_heads=8,
            )
            layers.append(layer)
        
        self.layers = nn.ModuleList(layers)
        
        # Calculate final spatial size
        final_size = input_size
        for i in range(num_layers):
            if i < num_layers - 2:  # Layers with stride 2
                final_size = final_size // 2
        
        # Final classification head
        final_channels = channels[-1]
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(final_channels, final_channels // 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Linear(final_channels // 2, 1)),
        )
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Forward pass for image quality branch.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Dictionary containing quality scores and intermediate features
        """
        features = []
        
        # Forward through layers
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        
        # Global pooling and classification
        pooled = self.global_pool(x)  # (B, C, 1, 1)
        pooled = pooled.view(pooled.size(0), -1)  # (B, C)
        
        quality_score = self.classifier(pooled)  # (B, 1)
        
        return {
            'quality_score': quality_score,
            'features': features,
            'final_feature': pooled,
        }


class ConsistencyBranch(nn.Module):
    """
    3D Consistency Branch of the discriminator.
    
    This branch evaluates 3D geometric consistency across different viewpoints.
    Uses multi-view analysis to ensure generated content maintains 3D coherence.
    """
    
    def __init__(
        self,
        input_size: int = 256,
        input_channels: int = 3,
        feature_dim: int = 256,
        num_views: int = 2,  # Number of views to compare
        use_cross_attention: bool = True,
    ):
        """
        Initialize consistency branch.
        
        Args:
            input_size: Input image size
            input_channels: Number of input channels
            feature_dim: Feature dimensionality
            num_views: Number of views for consistency analysis
            use_cross_attention: Whether to use cross-attention between views
        """
        super().__init__()
        
        self.num_views = num_views
        self.feature_dim = feature_dim
        self.use_cross_attention = use_cross_attention
        
        # Feature extractor (shared across views)
        self.feature_extractor = nn.Sequential(
            DiscriminatorBlock(input_channels, 64, stride=2),
            DiscriminatorBlock(64, 128, stride=2),
            DiscriminatorBlock(128, 256, stride=2, use_attention=True),
            DiscriminatorBlock(256, feature_dim, stride=2),
            nn.AdaptiveAvgPool2d(4),  # Output: (B, feature_dim, 4, 4)
        )
        
        # Cross-view attention for consistency analysis
        if use_cross_attention:
            # Add a projection layer to match attention input dimensions
            flattened_dim = feature_dim * 16  # 4x4 spatial features flattened
            self.feature_projection = nn.Linear(flattened_dim, feature_dim)
            
            self.cross_attention = MultiHeadSelfAttention(
                dim=feature_dim,
                num_heads=8,
                qkv_bias=True,
                attn_drop=0.1,
                proj_drop=0.1,
            )
        
        # Consistency scorer
        consistency_input_dim = feature_dim * 16  # 4x4 spatial features
        if use_cross_attention:
            consistency_input_dim *= 2  # Original + cross-attended features
        
        self.consistency_scorer = nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(consistency_input_dim, feature_dim)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Linear(feature_dim, feature_dim // 2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(nn.Linear(feature_dim // 2, 1)),
        )
    
    def forward(self, views: List[Tensor]) -> Dict[str, Tensor]:
        """
        Forward pass for consistency branch.
        
        Args:
            views: List of view images, each (B, C, H, W)
            
        Returns:
            Dictionary containing consistency scores and features
        """
        if len(views) != self.num_views:
            raise ValueError(f"Expected {self.num_views} views, got {len(views)}")
        
        # Extract features from each view
        view_features = []
        for view in views:
            features = self.feature_extractor(view)  # (B, feature_dim, 4, 4)
            features = features.view(features.size(0), -1)  # (B, feature_dim * 16)
            view_features.append(features)
        
        # Stack view features for cross-view analysis
        stacked_features = torch.stack(view_features, dim=1)  # (B, num_views, feature_dim * 16)
        
        # Cross-view attention (optional)
        if self.use_cross_attention:
            # Project to attention dimension
            projected_features = self.feature_projection(stacked_features)  # (B, num_views, feature_dim)
            attended_features = self.cross_attention(projected_features)  # (B, num_views, feature_dim)
            # Project back to original dimension
            attended_features = F.linear(attended_features, self.feature_projection.weight.t(), None)  # (B, num_views, feature_dim * 16)
            # Concatenate original and attended features
            combined_features = torch.cat([stacked_features, attended_features], dim=-1)
        else:
            combined_features = stacked_features
        
        # Aggregate across views (mean pooling)
        aggregated = combined_features.mean(dim=1)  # (B, feature_dim * 16 * [1 or 2])
        
        # Compute consistency score
        consistency_score = self.consistency_scorer(aggregated)  # (B, 1)
        
        return {
            'consistency_score': consistency_score,
            'view_features': view_features,
            'aggregated_feature': aggregated,
        }


class AetheristDiscriminator(nn.Module):
    """
    Main discriminator architecture for Aetherist.
    
    This discriminator uses a dual-branch architecture to evaluate both image quality
    and 3D consistency. It provides detailed feedback for training the generator.
    
    Architecture Components:
    1. Image Quality Branch: Evaluates photorealism and visual quality
    2. 3D Consistency Branch: Assesses geometric consistency across viewpoints
    3. Multi-scale Analysis: Processes images at different resolutions
    4. Spectral Normalization: Ensures training stability
    """
    
    def __init__(
        self,
        # Input configuration
        input_size: int = 256,
        input_channels: int = 3,
        
        # Architecture configuration
        base_channels: int = 64,
        feature_dim: int = 256,
        
        # Multi-scale configuration
        use_multiscale: bool = True,
        scales: List[int] = None,
        
        # Consistency analysis
        num_views: int = 2,
        use_consistency_branch: bool = True,
        
        # Training configuration
        use_gradient_penalty: bool = True,
        lambda_consistency: float = 0.1,
    ):
        """
        Initialize Aetherist discriminator.
        
        Args:
            input_size: Input image size
            input_channels: Number of input channels
            base_channels: Base number of channels
            feature_dim: Feature dimensionality
            use_multiscale: Whether to use multi-scale analysis
            scales: List of scales for multi-scale analysis
            num_views: Number of views for consistency analysis
            use_consistency_branch: Whether to use 3D consistency branch
            use_gradient_penalty: Whether to use gradient penalty
            lambda_consistency: Weight for consistency loss
        """
        super().__init__()
        
        self.input_size = input_size
        self.use_multiscale = use_multiscale
        self.use_consistency_branch = use_consistency_branch
        self.lambda_consistency = lambda_consistency
        self.num_views = num_views
        
        if scales is None:
            scales = [256, 128, 64] if use_multiscale else [input_size]
        self.scales = scales
        
        # Image Quality Branch
        self.quality_branch = ImageQualityBranch(
            input_size=input_size,
            input_channels=input_channels,
            base_channels=base_channels,
            max_channels=512,
            num_layers=6,
        )
        
        # 3D Consistency Branch (optional)
        if use_consistency_branch:
            self.consistency_branch = ConsistencyBranch(
                input_size=input_size,
                input_channels=input_channels,
                feature_dim=feature_dim,
                num_views=num_views,
                use_cross_attention=True,
            )
        
        # Multi-scale quality branches (optional)
        if use_multiscale and len(scales) > 1:
            self.multiscale_branches = nn.ModuleList([
                ImageQualityBranch(
                    input_size=scale,
                    input_channels=input_channels,
                    base_channels=base_channels,
                    num_layers=4,  # Fewer layers for smaller scales
                )
                for scale in scales[1:]  # Skip the main scale
            ])
        
        # Final discriminator head - we'll build this dynamically
        # since input size depends on whether consistency analysis is used
        self.feature_dim = feature_dim
        self.quality_feature_dim = 512  # From ImageQualityBranch
        
        # We'll create the final head in the forward pass based on actual feature size
        self._final_heads = {}  # Cache for different input sizes
    
    def _get_final_head(self, input_dim: int) -> nn.Module:
        """Get or create a final head for the given input dimension."""
        if input_dim not in self._final_heads:
            final_head = nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(input_dim, self.feature_dim)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.utils.spectral_norm(nn.Linear(self.feature_dim, self.feature_dim // 2)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.utils.spectral_norm(nn.Linear(self.feature_dim // 2, 1)),
            ).to(next(self.parameters()).device)
            self._final_heads[input_dim] = final_head
        
        return self._final_heads[input_dim]
    
    def forward(
        self,
        images: Union[Tensor, List[Tensor]],
        return_features: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Forward pass of the discriminator.
        
        Args:
            images: Input images. Can be:
                - Single tensor (B, C, H, W) for quality-only analysis
                - List of tensors for multi-view consistency analysis
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing discriminator outputs
        """
        # Handle different input formats
        if isinstance(images, list):
            if len(images) != self.num_views:
                raise ValueError(f"Expected {self.num_views} views, got {len(images)}")
            main_image = images[0]
            all_views = images
        else:
            main_image = images
            all_views = [images]
        
        results = {}
        features_list = []
        
        # 1. Image Quality Analysis
        quality_output = self.quality_branch(main_image)
        results['quality_score'] = quality_output['quality_score']
        features_list.append(quality_output['final_feature'])
        
        if return_features:
            results['quality_features'] = quality_output['features']
        
        # 2. 3D Consistency Analysis (if enabled and multiple views provided)
        if self.use_consistency_branch and len(all_views) >= self.num_views:
            consistency_output = self.consistency_branch(all_views[:self.num_views])
            results['consistency_score'] = consistency_output['consistency_score']
            features_list.append(consistency_output['aggregated_feature'])
            
            if return_features:
                results['consistency_features'] = consistency_output['view_features']
        
        # 3. Multi-scale Analysis (optional)
        if self.use_multiscale and hasattr(self, 'multiscale_branches'):
            multiscale_scores = []
            for i, (scale, branch) in enumerate(zip(self.scales[1:], self.multiscale_branches)):
                # Resize image to target scale
                resized = F.interpolate(main_image, size=(scale, scale), mode='bilinear', align_corners=False)
                scale_output = branch(resized)
                multiscale_scores.append(scale_output['quality_score'])
                
                if return_features:
                    results[f'scale_{scale}_features'] = scale_output['features']
            
            results['multiscale_scores'] = multiscale_scores
        
        # 4. Final Prediction
        if len(features_list) > 1:
            combined_features = torch.cat(features_list, dim=1)
        else:
            combined_features = features_list[0]
        
        # Get appropriate final head for the feature dimension
        final_head = self._get_final_head(combined_features.shape[1])
        final_score = final_head(combined_features)
        results['final_score'] = final_score
        
        # 5. Combined Loss (weighted)
        total_score = results['quality_score']
        if 'consistency_score' in results:
            total_score = total_score + self.lambda_consistency * results['consistency_score']
        
        results['total_score'] = total_score
        
        return results


def test_discriminator_components():
    """Test function for discriminator components."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 2
    input_size = 256
    input_channels = 3
    num_views = 2
    
    print("Testing Discriminator Components...")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Input size: {input_size}x{input_size}")
    print()
    
    # Test Image Quality Branch
    print("Testing Image Quality Branch...")
    quality_branch = ImageQualityBranch(
        input_size=input_size,
        input_channels=input_channels,
    ).to(device)
    
    test_image = torch.randn(batch_size, input_channels, input_size, input_size, device=device)
    quality_output = quality_branch(test_image)
    
    print(f"Input: {test_image.shape}")
    print(f"Quality score: {quality_output['quality_score'].shape}")
    print(f"Final feature: {quality_output['final_feature'].shape}")
    print(f"Number of feature maps: {len(quality_output['features'])}")
    print()
    
    # Test Consistency Branch
    print("Testing 3D Consistency Branch...")
    consistency_branch = ConsistencyBranch(
        input_size=input_size,
        input_channels=input_channels,
        num_views=num_views,
    ).to(device)
    
    test_views = [
        torch.randn(batch_size, input_channels, input_size, input_size, device=device)
        for _ in range(num_views)
    ]
    
    consistency_output = consistency_branch(test_views)
    
    print(f"Views: {[view.shape for view in test_views]}")
    print(f"Consistency score: {consistency_output['consistency_score'].shape}")
    print(f"Aggregated feature: {consistency_output['aggregated_feature'].shape}")
    print()
    
    # Test Full Discriminator
    print("Testing Full Discriminator...")
    discriminator = AetheristDiscriminator(
        input_size=input_size,
        input_channels=input_channels,
        use_multiscale=True,
        use_consistency_branch=True,
        num_views=num_views,
    ).to(device)
    
    # Test with single image
    single_output = discriminator(test_image, return_features=True)
    print("Single Image Analysis:")
    print(f"Quality score: {single_output['quality_score'].shape}")
    print(f"Final score: {single_output['final_score'].shape}")
    print(f"Total score: {single_output['total_score'].shape}")
    print()
    
    # Test with multiple views
    multi_output = discriminator(test_views, return_features=True)
    print("Multi-view Analysis:")
    print(f"Quality score: {multi_output['quality_score'].shape}")
    print(f"Consistency score: {multi_output['consistency_score'].shape}")
    print(f"Final score: {multi_output['final_score'].shape}")
    print(f"Total score: {multi_output['total_score'].shape}")
    print()
    
    # Parameter count
    total_params = sum(p.numel() for p in discriminator.parameters())
    trainable_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print()
    
    print("✅ All discriminator component tests passed!")


if __name__ == "__main__":
    test_discriminator_components()