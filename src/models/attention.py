"""
Attention mechanisms for Aetherist Vision Transformer.
Optimized attention blocks for large feature maps and 3D-aware generation.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import einops
from einops import rearrange, repeat


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention optimized for large feature maps.
    
    Implements efficient attention with optional memory optimizations:
    - Flash attention-style computation
    - Gradient checkpointing support
    - Memory-efficient attention for very large sequences
    
    Mathematical formulation:
    Attention(Q, K, V) = softmax(QK^T / √d_k)V
    MultiHead = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_flash_attention: bool = True,
        scale: Optional[float] = None,
    ):
        """
        Initialize multi-head self-attention.
        
        Args:
            dim: Input feature dimensionality
            num_heads: Number of attention heads
            qkv_bias: Whether to include bias in QKV projections
            attn_drop: Attention dropout probability
            proj_drop: Output projection dropout probability
            use_flash_attention: Whether to use memory-efficient attention
            scale: Custom attention scale (defaults to 1/sqrt(head_dim))
        """
        super().__init__()
        
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = scale or self.head_dim ** -0.5
        self.use_flash_attention = use_flash_attention
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Attention dropout
        self.attn_drop = nn.Dropout(attn_drop)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(
        self, 
        x: Tensor, 
        mask: Optional[Tensor] = None,
        return_attention: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass of multi-head self-attention.
        
        Args:
            x: Input tensor (B, N, C) where N is sequence length
            mask: Optional attention mask (B, N, N) or (N, N)
            return_attention: Whether to return attention weights
            
        Returns:
            Output tensor (B, N, C) or tuple of (output, attention_weights)
        """
        B, N, C = x.shape
        
        # Compute QKV
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each: (B, num_heads, N, head_dim)
        
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's native flash attention if available
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                scale=self.scale,
            )
            attn_weights = None  # Flash attention doesn't return weights
        else:
            # Standard attention computation
            attn_weights = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
            
            # Apply mask if provided
            if mask is not None:
                if mask.dim() == 2:
                    mask = mask.unsqueeze(0).unsqueeze(1)  # (1, 1, N, N)
                elif mask.dim() == 3:
                    mask = mask.unsqueeze(1)  # (B, 1, N, N)
                
                attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attn_drop(attn_weights)
            
            attn_output = attn_weights @ v  # (B, num_heads, N, head_dim)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        output = self.proj(attn_output)
        output = self.proj_drop(output)
        
        if return_attention and attn_weights is not None:
            return output, attn_weights
        else:
            return output


class CrossAttention(nn.Module):
    """
    Cross-attention for conditioning on external information.
    Used for text conditioning or cross-modal interactions.
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: Optional[int] = None,
        value_dim: Optional[int] = None,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        scale: Optional[float] = None,
    ):
        """
        Initialize cross-attention module.
        
        Args:
            query_dim: Query feature dimensionality
            key_dim: Key feature dimensionality (defaults to query_dim)
            value_dim: Value feature dimensionality (defaults to key_dim)
            num_heads: Number of attention heads
            qkv_bias: Whether to include bias in projections
            attn_drop: Attention dropout probability
            proj_drop: Output projection dropout probability
            scale: Custom attention scale
        """
        super().__init__()
        
        key_dim = key_dim or query_dim
        value_dim = value_dim or key_dim
        
        assert query_dim % num_heads == 0, f"query_dim {query_dim} must be divisible by num_heads {num_heads}"
        
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = scale or self.head_dim ** -0.5
        
        # Separate projections for Q, K, V
        self.q_proj = nn.Linear(query_dim, query_dim, bias=qkv_bias)
        self.k_proj = nn.Linear(key_dim, query_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(value_dim, query_dim, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(query_dim, query_dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor] = None,
        value: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass of cross-attention.
        
        Args:
            query: Query tensor (B, N_q, C_q)
            key: Key tensor (B, N_k, C_k), defaults to query
            value: Value tensor (B, N_v, C_v), defaults to key
            mask: Optional attention mask
            
        Returns:
            Output tensor (B, N_q, C_q)
        """
        if key is None:
            key = query
        if value is None:
            value = key
        
        B, N_q, _ = query.shape
        N_k = key.shape[1]
        
        # Project to Q, K, V
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N_k, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N_q, N_k)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        out = attn @ v  # (B, num_heads, N_q, head_dim)
        out = out.transpose(1, 2).reshape(B, N_q, self.query_dim)
        
        # Final projection
        out = self.proj(out)
        out = self.proj_drop(out)
        
        return out


class WindowAttention(nn.Module):
    """
    Window-based attention for efficient processing of large feature maps.
    
    Divides the input into windows and applies attention within each window,
    significantly reducing computational complexity for large sequences.
    """
    
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int] = (7, 7),
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """
        Initialize window attention.
        
        Args:
            dim: Input feature dimensionality
            window_size: Size of attention windows (H, W)
            num_heads: Number of attention heads
            qkv_bias: Whether to include bias in QKV projections
            attn_drop: Attention dropout probability
            proj_drop: Output projection dropout probability
        """
        super().__init__()
        
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # Get relative position indices
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))  # (2, Wh, Ww)
        coords_flatten = torch.flatten(coords, 1)  # (2, Wh*Ww)
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # (2, Wh*Ww, Wh*Ww)
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # (Wh*Ww, Wh*Ww, 2)
        relative_coords[:, :, 0] += window_size[0] - 1  # Shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        
        relative_position_index = relative_coords.sum(-1)  # (Wh*Ww, Wh*Ww)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Initialize relative position bias
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
    
    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass of window attention.
        
        Args:
            x: Input tensor (B, H, W, C)
            mask: Optional attention mask
            
        Returns:
            Output tensor (B, H, W, C)
        """
        B, H, W, C = x.shape
        
        # Partition windows
        x_windows = self.window_partition(x)  # (B*num_windows, window_size*window_size, C)
        
        # Apply attention
        attn_windows = self.attention_forward(x_windows, mask)  # (B*num_windows, window_size*window_size, C)
        
        # Merge windows
        x = self.window_reverse(attn_windows, H, W)  # (B, H, W, C)
        
        return x
    
    def window_partition(self, x: Tensor) -> Tensor:
        """Partition input into windows."""
        B, H, W, C = x.shape
        window_h, window_w = self.window_size
        
        x = x.view(B, H // window_h, window_h, W // window_w, window_w, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_h * window_w, C)
        
        return windows
    
    def window_reverse(self, windows: Tensor, H: int, W: int) -> Tensor:
        """Reverse window partitioning."""
        window_h, window_w = self.window_size
        B = int(windows.shape[0] / (H * W / window_h / window_w))
        
        x = windows.view(B, H // window_h, W // window_w, window_h, window_w, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        
        return x
    
    def attention_forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Apply attention within windows."""
        B_windows, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B_windows, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        
        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        )  # (Wh*Ww, Wh*Ww, nH)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (nH, Wh*Ww, Wh*Ww)
        attn = attn + relative_position_bias.unsqueeze(0)
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        x = attn @ v
        x = x.transpose(1, 2).reshape(B_windows, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


def test_attention_mechanisms():
    """Test function for attention mechanisms."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 2
    seq_len = 64
    feature_dim = 512
    num_heads = 8
    
    print("Testing Multi-Head Self-Attention...")
    
    # Test standard self-attention
    attention = MultiHeadSelfAttention(
        dim=feature_dim,
        num_heads=num_heads,
        use_flash_attention=False
    ).to(device)
    
    x = torch.randn(batch_size, seq_len, feature_dim, device=device)
    out = attention(x)
    
    print(f"Self-attention input: {x.shape}, output: {out.shape}")
    assert out.shape == x.shape
    
    # Test with attention weights
    out, attn_weights = attention(x, return_attention=True)
    print(f"Attention weights shape: {attn_weights.shape}")
    assert attn_weights.shape == (batch_size, num_heads, seq_len, seq_len)
    
    print("Testing Cross-Attention...")
    
    # Test cross-attention
    cross_attn = CrossAttention(
        query_dim=feature_dim,
        key_dim=feature_dim,
        num_heads=num_heads
    ).to(device)
    
    query = torch.randn(batch_size, seq_len, feature_dim, device=device)
    key = torch.randn(batch_size, seq_len // 2, feature_dim, device=device)
    
    out = cross_attn(query, key)
    print(f"Cross-attention query: {query.shape}, key: {key.shape}, output: {out.shape}")
    assert out.shape == query.shape
    
    print("Testing Window Attention...")
    
    # Test window attention (needs 2D input)
    H, W = 14, 14
    window_attn = WindowAttention(
        dim=feature_dim,
        window_size=(7, 7),
        num_heads=num_heads
    ).to(device)
    
    x_2d = torch.randn(batch_size, H, W, feature_dim, device=device)
    out_2d = window_attn(x_2d)
    
    print(f"Window attention input: {x_2d.shape}, output: {out_2d.shape}")
    assert out_2d.shape == x_2d.shape
    
    print("All attention mechanism tests passed!")


if __name__ == "__main__":
    test_attention_mechanisms()