"""
Neural network layers for Aetherist.
Contains custom layers for modulation, attention, and specialized operations.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import einops


class AdaptiveLayerNorm(nn.Module):
    """
    Adaptive Layer Normalization (AdaLN) for style modulation.
    
    AdaLN modulates the normalization parameters using style vectors,
    allowing for fine-grained control over feature distributions.
    
    Mathematical formulation:
    AdaLN(x, w) = γ(w) * LayerNorm(x) + β(w)
    
    Where:
    - LayerNorm(x) = (x - μ) / σ
    - γ(w) and β(w) are learned transformations of the style code w
    - μ and σ are the mean and standard deviation of x
    """
    
    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        style_dim: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        use_bias: bool = True,
    ):
        """
        Initialize Adaptive Layer Normalization.
        
        Args:
            normalized_shape: Shape of the normalized features
            style_dim: Dimensionality of the style vector
            eps: Small constant for numerical stability
            elementwise_affine: Whether to learn affine parameters
            use_bias: Whether to use bias in style transformations
        """
        super().__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        
        self.normalized_shape = normalized_shape
        self.style_dim = style_dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        # Compute the number of features
        self.num_features = 1
        for dim in normalized_shape:
            self.num_features *= dim
        
        if elementwise_affine:
            # Style-to-scale transformation
            self.style_to_scale = nn.Linear(style_dim, self.num_features, bias=use_bias)
            # Style-to-shift transformation  
            self.style_to_shift = nn.Linear(style_dim, self.num_features, bias=use_bias)
            
            # Initialize scale to 1 and shift to 0
            nn.init.ones_(self.style_to_scale.weight)
            nn.init.zeros_(self.style_to_shift.weight)
            
            if use_bias:
                nn.init.zeros_(self.style_to_scale.bias)
                nn.init.zeros_(self.style_to_shift.bias)
        else:
            self.register_parameter('style_to_scale', None)
            self.register_parameter('style_to_shift', None)
    
    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        """
        Apply adaptive layer normalization.
        
        Args:
            x: Input tensor (..., *normalized_shape)
            style: Style vector (B, style_dim)
            
        Returns:
            Modulated and normalized tensor
        """
        # Compute layer norm statistics
        # Keep all but the last len(normalized_shape) dimensions for mean/var computation
        reduce_dims = list(range(-len(self.normalized_shape), 0))
        
        mean = x.mean(dim=reduce_dims, keepdim=True)
        var = x.var(dim=reduce_dims, keepdim=True, unbiased=False)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        if self.elementwise_affine:
            # Compute modulation parameters from style
            scale = self.style_to_scale(style)  # (B, num_features)
            shift = self.style_to_shift(style)  # (B, num_features)
            
            # Reshape scale and shift to match normalized dimensions
            # For input (B, N, D) with normalized_shape (D,), we want (B, 1, D)
            if len(x.shape) == 3 and len(self.normalized_shape) == 1:
                # Token sequence case: (B, N, D) -> scale/shift should be (B, 1, D) 
                scale = scale.unsqueeze(1)  # (B, D) -> (B, 1, D)
                shift = shift.unsqueeze(1)  # (B, D) -> (B, 1, D)
            else:
                # General case: add dimensions to match x's shape
                for _ in range(len(x.shape) - len(style.shape)):
                    scale = scale.unsqueeze(-1)
                    shift = shift.unsqueeze(-1)
                
                # Reshape to match normalized_shape
                new_shape = [scale.size(0)] + list(self.normalized_shape) + [1] * (len(x.shape) - len(self.normalized_shape) - 1)
                scale = scale.view(new_shape)
                shift = shift.view(new_shape)
            
            # Apply modulation
            return scale * x_norm + shift
        else:
            return x_norm
    
    def extra_repr(self) -> str:
        return f"{self.normalized_shape}, style_dim={self.style_dim}, eps={self.eps}, " \
               f"elementwise_affine={self.elementwise_affine}"


class ModulatedLinear(nn.Module):
    """
    Linear layer with style modulation.
    
    Applies style-based weight modulation as in StyleGAN.
    w' = w * (s + 1) where s is the style modulation.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        style_dim: int,
        demodulate: bool = True,
        bias: bool = True,
        lr_multiplier: float = 1.0,
    ):
        """
        Initialize modulated linear layer.
        
        Args:
            in_features: Input feature dimensionality
            out_features: Output feature dimensionality
            style_dim: Style vector dimensionality
            demodulate: Whether to apply weight demodulation
            bias: Whether to include bias
            lr_multiplier: Learning rate multiplier
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.style_dim = style_dim
        self.demodulate = demodulate
        self.lr_multiplier = lr_multiplier
        
        # Weight and bias parameters
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Style modulation
        self.style_modulation = nn.Linear(style_dim, in_features, bias=True)
        nn.init.ones_(self.style_modulation.weight)
        nn.init.zeros_(self.style_modulation.bias)
        
        # Scaling factor for equalized learning rates
        self.scale = lr_multiplier / math.sqrt(in_features)
    
    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        """
        Forward pass with style modulation.
        
        Args:
            x: Input tensor (B, *, in_features)
            style: Style vector (B, style_dim)
            
        Returns:
            Modulated output (B, *, out_features)
        """
        batch_size = x.size(0)
        
        # Get style modulation
        s = self.style_modulation(style)  # (B, in_features)
        
        # Apply runtime scaling
        weight = self.weight * self.scale  # (out_features, in_features)
        
        # Modulate weights: w' = w * (s + 1)
        # Reshape s to (B, 1, in_features) for broadcasting
        s = s.view(batch_size, 1, -1)
        weight = weight.unsqueeze(0)  # (1, out_features, in_features)
        modulated_weight = weight * (s + 1)  # (B, out_features, in_features)
        
        if self.demodulate:
            # Compute demodulation factor
            # sigma = sqrt(sum(w'^2))
            demod = torch.rsqrt(torch.sum(modulated_weight ** 2, dim=2, keepdim=True) + 1e-8)
            modulated_weight = modulated_weight * demod
        
        # Reshape input for batch matrix multiplication
        x_shape = x.shape
        x = x.view(batch_size, -1, self.in_features)  # (B, N, in_features)
        
        # Apply modulated transformation
        # out = x @ w'.transpose(-1, -2)
        out = torch.bmm(x, modulated_weight.transpose(-1, -2))  # (B, N, out_features)
        
        # Add bias if present
        if self.bias is not None:
            out = out + self.bias
        
        # Reshape back to original shape
        out = out.view(*x_shape[:-1], self.out_features)
        
        return out


class SineLayer(nn.Module):
    """
    Sine activation layer for SIREN networks.
    
    SIREN (Sinusoidal Representation Networks) use sine activations
    to capture high-frequency details effectively.
    
    Mathematical formulation:
    SIREN(x) = sin(ω * (Wx + b))
    
    Where ω is the frequency parameter.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ):
        """
        Initialize SIREN layer.
        
        Args:
            in_features: Input feature dimensionality
            out_features: Output feature dimensionality
            bias: Whether to include bias
            is_first: Whether this is the first layer (different initialization)
            omega_0: Frequency parameter
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights according to SIREN paper."""
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform distribution
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                # Hidden layers: uniform distribution scaled by omega_0
                bound = math.sqrt(6 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with sine activation.
        
        Args:
            x: Input tensor
            
        Returns:
            Sine-activated output
        """
        return torch.sin(self.omega_0 * self.linear(x))


class FusedMLP(nn.Module):
    """
    Fused MLP block optimized for performance.
    
    Combines linear transformation, activation, and optional modulation
    in an efficient implementation.
    """
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        style_dim: Optional[int] = None,
        activation: str = "gelu",
        dropout_prob: float = 0.0,
        use_bias: bool = True,
    ):
        """
        Initialize fused MLP.
        
        Args:
            in_features: Input feature dimensionality
            hidden_features: Hidden layer dimensionality (defaults to 4 * in_features)
            out_features: Output dimensionality (defaults to in_features)
            style_dim: Style vector dimensionality (None = no modulation)
            activation: Activation function name
            dropout_prob: Dropout probability
            use_bias: Whether to use bias
        """
        super().__init__()
        
        hidden_features = hidden_features or 4 * in_features
        out_features = out_features or in_features
        
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.use_modulation = style_dim is not None
        
        # First linear layer
        if self.use_modulation:
            self.fc1 = ModulatedLinear(in_features, hidden_features, style_dim, bias=use_bias)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features, bias=use_bias)
        
        # Activation
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU(inplace=True)
        elif activation == "swish":
            self.activation = nn.SiLU(inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Dropout
        if dropout_prob > 0:
            self.dropout = nn.Dropout(dropout_prob)
        else:
            self.dropout = nn.Identity()
        
        # Second linear layer
        if self.use_modulation:
            self.fc2 = ModulatedLinear(hidden_features, out_features, style_dim, bias=use_bias)
        else:
            self.fc2 = nn.Linear(hidden_features, out_features, bias=use_bias)
    
    def forward(self, x: Tensor, style: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass through fused MLP.
        
        Args:
            x: Input tensor
            style: Optional style vector for modulation
            
        Returns:
            Output tensor
        """
        if self.use_modulation:
            if style is None:
                raise ValueError("Style vector required when using modulation")
            x = self.fc1(x, style)
        else:
            x = self.fc1(x)
        
        x = self.activation(x)
        x = self.dropout(x)
        
        if self.use_modulation:
            x = self.fc2(x, style)
        else:
            x = self.fc2(x)
        
        return x


def test_adaptive_layers():
    """Test function for adaptive layers."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    seq_len = 16
    feature_dim = 512
    style_dim = 256
    
    # Test AdaLN
    print("Testing Adaptive Layer Normalization...")
    adaln = AdaptiveLayerNorm(feature_dim, style_dim).to(device)
    
    x = torch.randn(batch_size, seq_len, feature_dim, device=device)
    style = torch.randn(batch_size, style_dim, device=device)
    
    out = adaln(x, style)
    print(f"AdaLN input shape: {x.shape}, output shape: {out.shape}")
    assert out.shape == x.shape
    
    # Test ModulatedLinear
    print("Testing Modulated Linear...")
    mod_linear = ModulatedLinear(feature_dim, feature_dim, style_dim).to(device)
    
    out = mod_linear(x, style)
    print(f"ModLinear input shape: {x.shape}, output shape: {out.shape}")
    assert out.shape == x.shape
    
    # Test SineLayer
    print("Testing SIREN Layer...")
    siren = SineLayer(feature_dim, feature_dim, is_first=True).to(device)
    
    out = siren(x)
    print(f"SIREN input shape: {x.shape}, output shape: {out.shape}")
    assert out.shape == x.shape
    
    # Test FusedMLP
    print("Testing Fused MLP...")
    mlp = FusedMLP(feature_dim, style_dim=style_dim).to(device)
    
    out = mlp(x, style)
    print(f"MLP input shape: {x.shape}, output shape: {out.shape}")
    assert out.shape == x.shape
    
    print("All layer tests passed!")


if __name__ == "__main__":
    test_adaptive_layers()