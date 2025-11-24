"""
Mathematical operations and signal processing utilities for Aetherist.
Implements alias-free signal processing operations inspired by StyleGAN3 for high-quality synthesis.
"""

import math
import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


def sinc_filter_1d(
    x: Tensor,
    cutoff: float = 0.5,
    window_size: int = 64,
    beta: float = 8.0
) -> Tensor:
    """
    Apply 1D sinc filter for anti-aliasing.
    
    The sinc filter is the ideal low-pass filter in the frequency domain.
    sinc(x) = sin(πx) / (πx)
    
    Args:
        x: Input tensor of shape (..., length)
        cutoff: Cutoff frequency as fraction of Nyquist frequency (0.5 = Nyquist)
        window_size: Size of the sinc filter kernel
        beta: Kaiser window parameter for tapering
        
    Returns:
        Filtered tensor of same shape as input
    """
    device = x.device
    dtype = x.dtype
    
    # Create sinc kernel
    n = torch.arange(-(window_size // 2), window_size // 2 + 1, device=device, dtype=dtype)
    
    # Sinc function: sin(π * cutoff * n) / (π * cutoff * n)
    # Handle n=0 case separately to avoid division by zero
    sinc_kernel = torch.zeros_like(n)
    non_zero = n != 0
    sinc_kernel[non_zero] = torch.sin(math.pi * cutoff * n[non_zero]) / (math.pi * cutoff * n[non_zero])
    sinc_kernel[n == 0] = cutoff  # lim_{n->0} sinc(cutoff * n) = cutoff
    
    # Apply Kaiser window for tapering
    kaiser_window = torch.kaiser_window(window_size + 1, beta=beta, device=device, dtype=dtype)
    sinc_kernel = sinc_kernel * kaiser_window
    
    # Normalize kernel
    sinc_kernel = sinc_kernel / sinc_kernel.sum()
    
    # Apply convolution
    padding = window_size // 2
    x_padded = F.pad(x, (padding, padding), mode='reflect')
    
    # Reshape for convolution
    original_shape = x.shape
    x_flat = x_padded.view(-1, 1, x_padded.size(-1))
    kernel = sinc_kernel.view(1, 1, -1)
    
    # Apply 1D convolution
    filtered = F.conv1d(x_flat, kernel, padding=0)
    
    # Reshape back to original shape
    return filtered.view(original_shape)


def sinc_filter_2d(
    x: Tensor,
    cutoff: float = 0.5,
    window_size: int = 64,
    beta: float = 8.0
) -> Tensor:
    """
    Apply 2D separable sinc filter for anti-aliasing.
    
    Args:
        x: Input tensor of shape (B, C, H, W)
        cutoff: Cutoff frequency as fraction of Nyquist frequency
        window_size: Size of the sinc filter kernel
        beta: Kaiser window parameter
        
    Returns:
        Filtered tensor of same shape as input
    """
    # Apply separable filtering: first along width, then height
    x_filtered_w = sinc_filter_1d(x, cutoff, window_size, beta)
    x_filtered_hw = sinc_filter_1d(x_filtered_w.transpose(-1, -2), cutoff, window_size, beta).transpose(-1, -2)
    
    return x_filtered_hw


class AliasFreeSampling(nn.Module):
    """
    Alias-free upsampling and downsampling operations.
    Implements the alias-free sampling from StyleGAN3 for artifact reduction.
    """
    
    def __init__(
        self,
        scale_factor: float,
        cutoff: float = 0.5,
        filter_size: int = 64,
        beta: float = 8.0,
    ):
        """
        Initialize alias-free sampling layer.
        
        Args:
            scale_factor: Sampling scale factor (>1 for upsampling, <1 for downsampling)
            cutoff: Filter cutoff frequency
            filter_size: Size of the anti-aliasing filter
            beta: Kaiser window parameter
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.cutoff = cutoff
        self.filter_size = filter_size
        self.beta = beta
        
        # Pre-compute filter kernel
        self.register_buffer('filter_kernel', self._create_filter_kernel())
    
    def _create_filter_kernel(self) -> Tensor:
        """Create the anti-aliasing filter kernel."""
        n = torch.arange(-(self.filter_size // 2), self.filter_size // 2 + 1, dtype=torch.float32)
        
        # Adjust cutoff for the scale factor
        effective_cutoff = self.cutoff / max(1, 1/self.scale_factor)
        
        # Create sinc kernel
        kernel = torch.zeros_like(n)
        non_zero = n != 0
        kernel[non_zero] = torch.sin(math.pi * effective_cutoff * n[non_zero]) / (math.pi * effective_cutoff * n[non_zero])
        kernel[n == 0] = effective_cutoff
        
        # Apply Kaiser window
        kaiser_window = torch.kaiser_window(self.filter_size + 1, beta=self.beta)
        kernel = kernel * kaiser_window
        
        # Normalize
        kernel = kernel / kernel.sum()
        
        return kernel
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Apply alias-free sampling.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Sampled tensor
        """
        if self.scale_factor == 1.0:
            return x
        
        # Apply anti-aliasing filter before resampling
        x_filtered = self._apply_filter_2d(x)
        
        # Perform resampling
        if self.scale_factor > 1:
            # Upsampling
            return F.interpolate(x_filtered, scale_factor=self.scale_factor, mode='linear', align_corners=False)
        else:
            # Downsampling
            return F.interpolate(x_filtered, scale_factor=self.scale_factor, mode='area')
    
    def _apply_filter_2d(self, x: Tensor) -> Tensor:
        """Apply 2D separable filtering."""
        # Apply filter along width
        padding = self.filter_size // 2
        x_padded = F.pad(x, (padding, padding, 0, 0), mode='reflect')
        
        # Reshape for convolution
        b, c, h, w_padded = x_padded.shape
        x_reshaped = x_padded.view(b * c * h, 1, w_padded)
        kernel = self.filter_kernel.view(1, 1, -1)
        
        # Apply 1D convolution along width
        x_filtered_w = F.conv1d(x_reshaped, kernel, padding=0)
        x_filtered_w = x_filtered_w.view(b, c, h, x_filtered_w.size(-1))
        
        # Apply filter along height
        x_padded = F.pad(x_filtered_w, (0, 0, padding, padding), mode='reflect')
        b, c, h_padded, w = x_padded.shape
        x_reshaped = x_padded.permute(0, 1, 3, 2).contiguous().view(b * c * w, 1, h_padded)
        
        # Apply 1D convolution along height
        x_filtered_h = F.conv1d(x_reshaped, kernel, padding=0)
        x_filtered_h = x_filtered_h.view(b, c, w, x_filtered_h.size(-1)).permute(0, 1, 3, 2)
        
        return x_filtered_h


def fused_leaky_relu(
    input: Tensor,
    bias: Optional[Tensor] = None,
    negative_slope: float = 0.2,
    scale: float = 2 ** 0.5,
) -> Tensor:
    """
    Fused leaky ReLU operation for improved performance.
    Combines bias addition and leaky ReLU activation in a single operation.
    
    Mathematical formulation:
    f(x) = scale * max(0, x + bias) + negative_slope * scale * min(0, x + bias)
    
    Args:
        input: Input tensor
        bias: Optional bias to add before activation
        negative_slope: Negative slope for leaky ReLU
        scale: Scaling factor for output
        
    Returns:
        Activated tensor with same shape as input
    """
    if bias is not None:
        input = input + bias.view(-1, *([1] * (input.dim() - 2)))
    
    return F.leaky_relu(input, negative_slope=negative_slope) * scale


def safe_normalize(
    x: Tensor,
    dim: int = -1,
    eps: float = 1e-8
) -> Tensor:
    """
    Safely normalize a tensor to unit norm.
    
    Args:
        x: Input tensor
        dim: Dimension along which to normalize
        eps: Small value to prevent division by zero
        
    Returns:
        Normalized tensor
    """
    norm = torch.norm(x, dim=dim, keepdim=True)
    return x / torch.clamp(norm, min=eps)


def slerp(
    a: Tensor,
    b: Tensor,
    t: Union[float, Tensor]
) -> Tensor:
    """
    Spherical linear interpolation between two tensors.
    
    SLERP formula:
    slerp(a, b, t) = (sin((1-t)θ) * a + sin(t*θ) * b) / sin(θ)
    where θ = arccos(a · b / (|a| * |b|))
    
    Args:
        a: Starting tensor
        b: Ending tensor  
        t: Interpolation parameter in [0, 1]
        
    Returns:
        Interpolated tensor
    """
    # Normalize inputs
    a_norm = safe_normalize(a.flatten(1), dim=1)
    b_norm = safe_normalize(b.flatten(1), dim=1)
    
    # Calculate angle between vectors
    dot_product = (a_norm * b_norm).sum(dim=1, keepdim=True)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    theta = torch.acos(dot_product)
    
    # Handle case where vectors are nearly parallel
    sin_theta = torch.sin(theta)
    use_lerp = sin_theta < 1e-6
    
    # SLERP calculation
    if isinstance(t, (int, float)):
        t = torch.full_like(sin_theta, t)
    
    sin_t_theta = torch.sin(t * theta)
    sin_one_minus_t_theta = torch.sin((1 - t) * theta)
    
    # Compute SLERP
    result_slerp = (sin_one_minus_t_theta * a_norm + sin_t_theta * b_norm) / sin_theta
    
    # Use linear interpolation for nearly parallel vectors
    result_lerp = (1 - t) * a_norm + t * b_norm
    result_lerp = safe_normalize(result_lerp, dim=1)
    
    # Choose between SLERP and LERP based on angle
    result = torch.where(use_lerp, result_lerp, result_slerp)
    
    # Reshape back to original shape
    return result.view_as(a)


def compute_gradient_penalty(
    discriminator: nn.Module,
    real_samples: Tensor,
    fake_samples: Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> Tensor:
    """
    Compute R1 gradient penalty for WGAN-GP style training.
    
    The R1 penalty is: λ/2 * E[||∇_x D(x)||²]
    where x ~ p_data(x)
    
    Args:
        discriminator: Discriminator network
        real_samples: Real data samples
        fake_samples: Generated samples (unused in R1, kept for interface compatibility)
        device: Device to compute on
        lambda_gp: Gradient penalty weight
        
    Returns:
        Gradient penalty loss
    """
    # R1 penalty only uses real samples
    batch_size = real_samples.size(0)
    
    # Enable gradient computation for real samples
    real_samples.requires_grad_(True)
    
    # Forward pass
    real_validity = discriminator(real_samples)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=real_validity,
        inputs=real_samples,
        grad_outputs=torch.ones_like(real_validity),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    # Compute gradient norm
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = lambda_gp * 0.5 * ((gradients.norm(2, dim=1)) ** 2).mean()
    
    return gradient_penalty