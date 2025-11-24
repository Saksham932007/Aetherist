"""
Mapping Network for Aetherist Generator.
Transforms random latent codes z to intermediate latent codes w for style-based generation.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MappingNetwork(nn.Module):
    """
    MLP Mapping Network that transforms latent codes z ∈ ℝ^512 to w ∈ ℝ^512.
    
    The mapping network serves several purposes:
    1. Disentanglement: Maps from a potentially entangled latent space Z to 
       a more disentangled intermediate space W
    2. Non-linearity: Provides multiple layers of non-linear transformations
    3. Conditioning: Allows for text/class conditioning through concatenation
    
    Mathematical formulation:
    w = f(z, c) where f is an 8-layer MLP with LeakyReLU activations
    Each layer: h_i+1 = LeakyReLU(W_i * h_i + b_i)
    """
    
    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 8,
        activation: str = "leaky_relu",
        dropout_prob: float = 0.0,
        normalize_input: bool = True,
        text_embedding_dim: int = 512,
        use_text_conditioning: bool = False,
        lr_multiplier: float = 0.01,
    ):
        """
        Initialize the mapping network.
        
        Args:
            latent_dim: Dimensionality of input latent code z
            hidden_dim: Hidden layer dimensionality
            num_layers: Number of MLP layers
            activation: Activation function ('leaky_relu', 'relu', 'gelu')
            dropout_prob: Dropout probability (0.0 = no dropout)
            normalize_input: Whether to apply pixel normalization to input
            text_embedding_dim: Dimensionality of text embeddings (if used)
            use_text_conditioning: Whether to use text conditioning
            lr_multiplier: Learning rate multiplier for the mapping network
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.normalize_input = normalize_input
        self.use_text_conditioning = use_text_conditioning
        self.lr_multiplier = lr_multiplier
        
        # Input dimensionality (latent + optional text conditioning)
        input_dim = latent_dim
        if use_text_conditioning:
            input_dim += text_embedding_dim
        
        # Create MLP layers
        layers = []
        
        # First layer
        layers.append(EqualizedLinear(input_dim, hidden_dim, lr_multiplier=lr_multiplier))
        layers.append(self._get_activation(activation))
        
        if dropout_prob > 0:
            layers.append(nn.Dropout(dropout_prob))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(EqualizedLinear(hidden_dim, hidden_dim, lr_multiplier=lr_multiplier))
            layers.append(self._get_activation(activation))
            
            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))
        
        self.mapping_layers = nn.Sequential(*layers)
        
        # Optional text projection layer
        if use_text_conditioning:
            self.text_projection = EqualizedLinear(
                text_embedding_dim, text_embedding_dim, lr_multiplier=lr_multiplier
            )
    
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name."""
        if activation == "leaky_relu":
            return nn.LeakyReLU(0.2, inplace=True)
        elif activation == "relu":
            return nn.ReLU(inplace=True)
        elif activation == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def pixel_norm(self, x: Tensor, epsilon: float = 1e-8) -> Tensor:
        """
        Apply pixel normalization.
        
        Pixel normalization: x / sqrt(mean(x^2) + ε)
        This helps with training stability in latent spaces.
        
        Args:
            x: Input tensor (..., features)
            epsilon: Small constant for numerical stability
            
        Returns:
            Normalized tensor
        """
        return x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + epsilon)
    
    def forward(
        self,
        z: Tensor,
        text_embeddings: Optional[Tensor] = None,
        truncation_psi: float = 1.0,
        truncation_cutoff: Optional[int] = None,
    ) -> Tensor:
        """
        Forward pass of the mapping network.
        
        Args:
            z: Random latent codes (B, latent_dim)
            text_embeddings: Optional text embeddings (B, text_embedding_dim)
            truncation_psi: Truncation parameter for style mixing (1.0 = no truncation)
            truncation_cutoff: Layer cutoff for truncation (None = all layers)
            
        Returns:
            Intermediate latent codes w (B, hidden_dim)
        """
        batch_size = z.size(0)
        
        # Input normalization
        if self.normalize_input:
            z = self.pixel_norm(z)
        
        # Prepare input
        x = z
        
        # Add text conditioning if available
        if self.use_text_conditioning:
            if text_embeddings is None:
                # Use zero embeddings if no text provided
                text_embeddings = torch.zeros(
                    batch_size, 
                    self.text_projection.in_features, 
                    device=z.device, 
                    dtype=z.dtype
                )
            else:
                # Project text embeddings
                text_embeddings = self.text_projection(text_embeddings)
            
            x = torch.cat([x, text_embeddings], dim=-1)
        
        # Apply mapping network
        w = self.mapping_layers(x)
        
        # Apply truncation if specified
        if truncation_psi < 1.0:
            w = self.apply_truncation(w, truncation_psi, truncation_cutoff)
        
        return w
    
    def apply_truncation(
        self,
        w: Tensor,
        truncation_psi: float,
        truncation_cutoff: Optional[int] = None,
    ) -> Tensor:
        """
        Apply truncation trick for controllable generation.
        
        Truncation interpolates between the generated w and the average w:
        w_truncated = w_avg + psi * (w - w_avg)
        
        Args:
            w: Latent codes (B, features)
            truncation_psi: Truncation parameter [0, 1]
            truncation_cutoff: Not used in mapping network (used in generator)
            
        Returns:
            Truncated latent codes
        """
        if not hasattr(self, 'w_avg') or self.w_avg.device != w.device:
            # Initialize w_avg as zeros if not computed yet
            self.register_buffer('w_avg', torch.zeros_like(w[0]))
        
        # Apply truncation
        w_truncated = self.w_avg.unsqueeze(0) + truncation_psi * (w - self.w_avg.unsqueeze(0))
        
        return w_truncated
    
    def compute_w_avg(self, num_samples: int = 10000, device: torch.device = torch.device("cpu")) -> Tensor:
        """
        Compute the average latent code w for truncation.
        
        Args:
            num_samples: Number of random samples to use
            device: Device to compute on
            
        Returns:
            Average latent code w_avg
        """
        self.eval()
        w_samples = []
        
        with torch.no_grad():
            for _ in range(0, num_samples, 100):  # Process in batches
                batch_size = min(100, num_samples - len(w_samples) * 100)
                z_batch = torch.randn(batch_size, self.latent_dim, device=device)
                w_batch = self.forward(z_batch, truncation_psi=1.0)
                w_samples.append(w_batch)
        
        w_avg = torch.cat(w_samples, dim=0).mean(dim=0)
        self.register_buffer('w_avg', w_avg)
        
        self.train()
        return w_avg


class EqualizedLinear(nn.Module):
    """
    Linear layer with equalized learning rates.
    
    Implements the "equalized learning rates" technique from StyleGAN.
    The weights are scaled by a runtime constant instead of using standard 
    initialization, ensuring all weights have the same learning rate.
    
    Mathematical formulation:
    y = (W / c) * x + b
    where c = sqrt(fan_in) and W ~ N(0, 1)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lr_multiplier: float = 1.0,
    ):
        """
        Initialize equalized linear layer.
        
        Args:
            in_features: Input feature dimensionality
            out_features: Output feature dimensionality  
            bias: Whether to include bias term
            lr_multiplier: Learning rate multiplier
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.lr_multiplier = lr_multiplier
        
        # Initialize weights from standard normal distribution
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Compute scaling factor for equalized learning rates
        # fan_in scaling ensures variance of output is approximately 1
        self.scale = lr_multiplier / math.sqrt(in_features)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass with runtime weight scaling.
        
        Args:
            x: Input tensor (..., in_features)
            
        Returns:
            Output tensor (..., out_features)
        """
        # Apply runtime scaling to weights
        weight = self.weight * self.scale
        
        # Linear transformation
        output = F.linear(x, weight, self.bias)
        
        return output
    
    def extra_repr(self) -> str:
        """String representation of the layer."""
        return f"in_features={self.in_features}, out_features={self.out_features}, " \
               f"bias={self.bias is not None}, lr_multiplier={self.lr_multiplier}"


def test_mapping_network():
    """Test function for the mapping network."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test basic mapping network
    mapping_net = MappingNetwork(
        latent_dim=512,
        hidden_dim=512,
        num_layers=8,
        use_text_conditioning=False
    ).to(device)
    
    # Test forward pass
    batch_size = 4
    z = torch.randn(batch_size, 512, device=device)
    w = mapping_net(z)
    
    print(f"Input shape: {z.shape}")
    print(f"Output shape: {w.shape}")
    assert w.shape == (batch_size, 512)
    
    # Test with text conditioning
    text_mapping_net = MappingNetwork(
        latent_dim=512,
        hidden_dim=512,
        num_layers=8,
        use_text_conditioning=True,
        text_embedding_dim=512
    ).to(device)
    
    text_embeddings = torch.randn(batch_size, 512, device=device)
    w_text = text_mapping_net(z, text_embeddings)
    
    print(f"Output with text conditioning: {w_text.shape}")
    assert w_text.shape == (batch_size, 512)
    
    # Test truncation
    mapping_net.compute_w_avg(num_samples=1000, device=device)
    w_truncated = mapping_net(z, truncation_psi=0.5)
    
    print(f"Truncated output shape: {w_truncated.shape}")
    assert w_truncated.shape == (batch_size, 512)
    
    print("All tests passed!")


if __name__ == "__main__":
    test_mapping_network()