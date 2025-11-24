"""
Vision Transformer blocks for Aetherist Generator.
Combines attention, MLP, and adaptive normalization for style-conditioned generation.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import einops
from einops import rearrange

from .attention import MultiHeadSelfAttention, CrossAttention
from .layers import AdaptiveLayerNorm, FusedMLP, SineLayer


class ViTBlock(nn.Module):
    """
    Vision Transformer block with style conditioning.
    
    Combines self-attention, cross-attention (for text), and MLP
    with adaptive layer normalization for style injection.
    
    Architecture:
    x = x + Attention(AdaLN(x, style))
    x = x + CrossAttention(AdaLN(x, style), text)  [optional]
    x = x + MLP(AdaLN(x, style))
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        style_dim: int = 512,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_drop: float = 0.0,
        use_cross_attention: bool = False,
        text_dim: Optional[int] = None,
        activation: str = "gelu",
        use_siren: bool = False,
        siren_omega: float = 30.0,
        pre_norm: bool = True,
    ):
        """
        Initialize ViT block.
        
        Args:
            dim: Feature dimensionality
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            style_dim: Style vector dimensionality
            qkv_bias: Whether to use bias in QKV projections
            attn_drop: Attention dropout probability
            proj_drop: Projection dropout probability
            mlp_drop: MLP dropout probability
            use_cross_attention: Whether to include cross-attention
            text_dim: Text embedding dimensionality (for cross-attention)
            activation: MLP activation function
            use_siren: Whether to use SIREN activations in MLP
            siren_omega: SIREN frequency parameter
            pre_norm: Whether to use pre-normalization (vs post-norm)
        """
        super().__init__()
        
        self.dim = dim
        self.style_dim = style_dim
        self.use_cross_attention = use_cross_attention
        self.pre_norm = pre_norm
        
        # Adaptive layer normalizations
        self.norm1 = AdaptiveLayerNorm(dim, style_dim)
        
        # Self-attention
        self.self_attn = MultiHeadSelfAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )
        
        # Cross-attention (optional)
        if use_cross_attention:
            self.norm_cross = AdaptiveLayerNorm(dim, style_dim)
            self.cross_attn = CrossAttention(
                query_dim=dim,
                key_dim=text_dim or dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
        else:
            self.norm_cross = None
            self.cross_attn = None
        
        # MLP
        self.norm2 = AdaptiveLayerNorm(dim, style_dim)
        
        if use_siren:
            # SIREN MLP for high-frequency details
            hidden_dim = int(dim * mlp_ratio)
            self.mlp = nn.Sequential(
                SineLayer(dim, hidden_dim, is_first=False, omega_0=siren_omega),
                nn.Dropout(mlp_drop),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(proj_drop),
            )
        else:
            # Standard MLP with style modulation
            self.mlp = FusedMLP(
                in_features=dim,
                hidden_features=int(dim * mlp_ratio),
                out_features=dim,
                style_dim=style_dim,
                activation=activation,
                dropout_prob=mlp_drop,
            )
        
        self.use_siren_mlp = use_siren
    
    def forward(
        self,
        x: Tensor,
        style: Tensor,
        text_embeddings: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass of ViT block.
        
        Args:
            x: Input features (B, N, C)
            style: Style vector (B, style_dim)
            text_embeddings: Optional text embeddings for cross-attention (B, T, text_dim)
            attention_mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Output features (B, N, C) or tuple with attention weights
        """
        attention_weights = None
        
        # Self-attention with adaptive normalization
        if self.pre_norm:
            # Pre-normalization: norm -> attention -> residual
            x_norm = self.norm1(x, style)
            if return_attention:
                attn_out, attention_weights = self.self_attn(
                    x_norm, mask=attention_mask, return_attention=True
                )
            else:
                attn_out = self.self_attn(x_norm, mask=attention_mask)
            x = x + attn_out
        else:
            # Post-normalization: attention -> residual -> norm
            if return_attention:
                attn_out, attention_weights = self.self_attn(
                    x, mask=attention_mask, return_attention=True
                )
            else:
                attn_out = self.self_attn(x, mask=attention_mask)
            x = self.norm1(x + attn_out, style)
        
        # Cross-attention (if enabled)
        if self.use_cross_attention and text_embeddings is not None:
            if self.pre_norm:
                x_norm_cross = self.norm_cross(x, style)
                cross_out = self.cross_attn(x_norm_cross, text_embeddings, text_embeddings)
                x = x + cross_out
            else:
                cross_out = self.cross_attn(x, text_embeddings, text_embeddings)
                x = self.norm_cross(x + cross_out, style)
        
        # MLP with adaptive normalization
        if self.pre_norm:
            x_norm2 = self.norm2(x, style)
            if self.use_siren_mlp:
                mlp_out = self.mlp(x_norm2)
            else:
                mlp_out = self.mlp(x_norm2, style)
            x = x + mlp_out
        else:
            if self.use_siren_mlp:
                mlp_out = self.mlp(x)
            else:
                mlp_out = self.mlp(x, style)
            x = self.norm2(x + mlp_out, style)
        
        if return_attention:
            return x, attention_weights
        else:
            return x


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer for converting images to sequence of patches.
    
    Converts input images into a sequence of patch embeddings that can be
    processed by transformer blocks.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        bias: bool = True,
    ):
        """
        Initialize patch embedding.
        
        Args:
            img_size: Input image size (assumed square)
            patch_size: Size of each patch
            in_channels: Number of input channels
            embed_dim: Embedding dimensionality
            bias: Whether to use bias in projection
        """
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch projection
        self.proj = nn.Conv2d(
            in_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size,
            bias=bias
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of patch embedding.
        
        Args:
            x: Input images (B, C, H, W)
            
        Returns:
            Patch embeddings (B, N, embed_dim) where N = (H/patch_size) * (W/patch_size)
        """
        B, C, H, W = x.shape
        
        # Check input size
        assert H == self.img_size and W == self.img_size, \
            f"Input size ({H}, {W}) doesn't match expected size ({self.img_size}, {self.img_size})"
        
        # Project patches
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        
        # Flatten patches
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Learnable positional encoding for transformer inputs.
    
    Supports both 1D (sequence) and 2D (spatial) positional encodings.
    """
    
    def __init__(
        self,
        embed_dim: int,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        encoding_type: str = "learned",
    ):
        """
        Initialize positional encoding.
        
        Args:
            embed_dim: Embedding dimensionality
            max_seq_len: Maximum sequence length
            dropout: Dropout probability
            encoding_type: Type of encoding ("learned" or "sinusoidal")
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.encoding_type = encoding_type
        
        if encoding_type == "learned":
            self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim) * 0.02)
        elif encoding_type == "sinusoidal":
            self.register_buffer("pos_embedding", self._create_sinusoidal_encoding())
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
        
        self.dropout = nn.Dropout(dropout)
    
    def _create_sinusoidal_encoding(self) -> Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(self.max_seq_len, self.embed_dim)
        position = torch.arange(0, self.max_seq_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float() * 
            (-math.log(10000.0) / self.embed_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # (1, max_seq_len, embed_dim)
    
    def forward(self, x: Tensor, seq_len: Optional[int] = None) -> Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (B, N, embed_dim)
            seq_len: Sequence length (defaults to x.size(1))
            
        Returns:
            Input with positional encoding added
        """
        if seq_len is None:
            seq_len = x.size(1)
        
        assert seq_len <= self.max_seq_len, \
            f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
        
        pos_emb = self.pos_embedding[:, :seq_len, :]
        x = x + pos_emb
        
        return self.dropout(x)


class VisionTransformer(nn.Module):
    """
    Vision Transformer backbone for the generator.
    
    Processes patch embeddings through multiple ViT blocks with style conditioning.
    """
    
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        style_dim: int = 512,
        use_cross_attention: bool = False,
        text_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_siren: bool = False,
        class_token: bool = False,
    ):
        """
        Initialize Vision Transformer.
        
        Args:
            img_size: Input image size
            patch_size: Patch size
            in_channels: Number of input channels
            embed_dim: Embedding dimensionality
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            style_dim: Style vector dimensionality
            use_cross_attention: Whether to use cross-attention
            text_dim: Text embedding dimensionality
            dropout: Dropout probability
            use_siren: Whether to use SIREN activations
            class_token: Whether to use a class token
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.use_class_token = class_token
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        
        num_patches = self.patch_embed.num_patches
        
        # Class token (optional)
        if class_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            num_tokens = num_patches + 1
        else:
            self.cls_token = None
            num_tokens = num_patches
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            embed_dim=embed_dim,
            max_seq_len=num_tokens,
            dropout=dropout,
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                style_dim=style_dim,
                attn_drop=dropout,
                proj_drop=dropout,
                mlp_drop=dropout,
                use_cross_attention=use_cross_attention,
                text_dim=text_dim,
                use_siren=use_siren,
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm = AdaptiveLayerNorm(embed_dim, style_dim)
    
    def forward(
        self,
        x: Tensor,
        style: Tensor,
        text_embeddings: Optional[Tensor] = None,
        return_attention: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, list]]:
        """
        Forward pass of Vision Transformer.
        
        Args:
            x: Input images (B, C, H, W)
            style: Style vectors (B, style_dim)
            text_embeddings: Optional text embeddings
            return_attention: Whether to return attention weights
            
        Returns:
            Feature representations or tuple with attention weights
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add class token if used
        if self.use_class_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer blocks
        attention_weights = []
        for block in self.blocks:
            if return_attention:
                x, attn = block(x, style, text_embeddings, return_attention=True)
                attention_weights.append(attn)
            else:
                x = block(x, style, text_embeddings)
        
        # Final normalization
        x = self.norm(x, style)
        
        if return_attention:
            return x, attention_weights
        else:
            return x


def test_vit_components():
    """Test function for ViT components."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 2
    img_size = 64
    patch_size = 8
    embed_dim = 256
    style_dim = 128
    
    print("Testing ViT Block...")
    
    # Test ViT block
    vit_block = ViTBlock(
        dim=embed_dim,
        num_heads=8,
        style_dim=style_dim,
        use_cross_attention=True,
        text_dim=embed_dim,
    ).to(device)
    
    seq_len = (img_size // patch_size) ** 2
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)
    style = torch.randn(batch_size, style_dim, device=device)
    text = torch.randn(batch_size, 16, embed_dim, device=device)
    
    out = vit_block(x, style, text)
    print(f"ViT Block input: {x.shape}, output: {out.shape}")
    assert out.shape == x.shape
    
    print("Testing Vision Transformer...")
    
    # Test full ViT
    vit = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        num_layers=4,
        style_dim=style_dim,
        use_cross_attention=True,
        text_dim=embed_dim,
    ).to(device)
    
    images = torch.randn(batch_size, 3, img_size, img_size, device=device)
    features = vit(images, style, text)
    
    expected_seq_len = seq_len
    print(f"ViT input: {images.shape}, output: {features.shape}")
    assert features.shape == (batch_size, expected_seq_len, embed_dim)
    
    print("All ViT component tests passed!")


if __name__ == "__main__":
    test_vit_components()