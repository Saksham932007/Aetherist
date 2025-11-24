"""
Text conditioning system for Aetherist.
Integrates CLIP text embeddings for text-to-3D generation.
"""

import math
from typing import Optional, Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import CrossAttention
from .layers import AdaptiveLayerNorm


class CLIPTextEncoder(nn.Module):
    """
    Frozen CLIP text encoder for generating text embeddings.
    
    Uses a pre-trained CLIP model to encode text prompts into
    high-dimensional embeddings that can condition the generator.
    """
    
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        freeze_weights: bool = True,
        max_length: int = 77,
    ):
        """
        Initialize CLIP text encoder.
        
        Args:
            model_name: CLIP model variant
            freeze_weights: Whether to freeze CLIP weights
            max_length: Maximum text sequence length
        """
        super().__init__()
        
        try:
            import open_clip
            self.clip_available = True
        except ImportError:
            print("Warning: open_clip not available, using dummy text encoder")
            self.clip_available = False
        
        self.max_length = max_length
        self.freeze_weights = freeze_weights
        
        if self.clip_available:
            # Load pre-trained CLIP model
            self.clip_model, _, _ = open_clip.create_model_and_transforms(model_name)
            self.tokenizer = open_clip.get_tokenizer(model_name)
            
            # Get text encoder and embedding dimension
            self.text_encoder = self.clip_model.text
            self.embedding_dim = self.clip_model.text.text_projection.out_features
            
            # Freeze weights if specified
            if freeze_weights:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
                self.text_encoder.eval()
        else:
            # Dummy implementation for testing
            self.embedding_dim = 512
            self.text_encoder = nn.Linear(max_length, self.embedding_dim)
            self.tokenizer = None
    
    def tokenize(self, texts: List[str]) -> Tensor:
        """
        Tokenize text prompts.
        
        Args:
            texts: List of text prompts
            
        Returns:
            Tokenized text tensor
        """
        if self.clip_available and self.tokenizer is not None:
            return self.tokenizer(texts)
        else:
            # Dummy tokenization
            batch_size = len(texts)
            return torch.randint(0, 1000, (batch_size, self.max_length))
    
    def forward(self, text_tokens: Tensor) -> Tensor:
        """
        Encode tokenized text to embeddings.
        
        Args:
            text_tokens: Tokenized text (B, max_length)
            
        Returns:
            Text embeddings (B, embedding_dim)
        """
        if self.clip_available:
            with torch.set_grad_enabled(not self.freeze_weights):
                # Get text features from CLIP
                text_features = self.text_encoder(text_tokens)
                # Normalize embeddings
                text_features = F.normalize(text_features, dim=-1)
                return text_features
        else:
            # Dummy implementation
            batch_size = text_tokens.size(0)
            return torch.randn(batch_size, self.embedding_dim, device=text_tokens.device)
    
    def encode_text(self, texts: List[str]) -> Tensor:
        """
        Convenience method to tokenize and encode text in one step.
        
        Args:
            texts: List of text prompts
            
        Returns:
            Text embeddings (B, embedding_dim)
        """
        tokens = self.tokenize(texts)
        if hasattr(tokens, 'to'):
            tokens = tokens.to(next(self.parameters()).device)
        return self.forward(tokens)


class TextConditioningModule(nn.Module):
    """
    Text conditioning module that injects text information into generator features.
    
    Uses cross-attention to condition visual features on text embeddings,
    enabling text-guided 3D generation.
    """
    
    def __init__(
        self,
        feature_dim: int,
        text_dim: int,
        num_heads: int = 8,
        num_layers: int = 2,
        style_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_style_modulation: bool = True,
    ):
        """
        Initialize text conditioning module.
        
        Args:
            feature_dim: Visual feature dimensionality
            text_dim: Text embedding dimensionality
            num_heads: Number of attention heads
            num_layers: Number of conditioning layers
            style_dim: Style vector dimensionality (for modulation)
            dropout: Dropout probability
            use_style_modulation: Whether to use style modulation
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.text_dim = text_dim
        self.num_layers = num_layers
        self.use_style_modulation = use_style_modulation
        
        # Text projection to match feature dimension
        self.text_projection = nn.Linear(text_dim, feature_dim)
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            CrossAttention(
                query_dim=feature_dim,
                key_dim=feature_dim,
                value_dim=feature_dim,
                num_heads=num_heads,
                attn_drop=dropout,
                proj_drop=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Normalization layers
        if use_style_modulation and style_dim is not None:
            self.norm_layers = nn.ModuleList([
                AdaptiveLayerNorm(feature_dim, style_dim)
                for _ in range(num_layers)
            ])
        else:
            self.norm_layers = nn.ModuleList([
                nn.LayerNorm(feature_dim)
                for _ in range(num_layers)
            ])
        
        # Feedforward layers
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim * 4, feature_dim),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        
        # Final projection
        self.final_projection = nn.Linear(feature_dim, feature_dim)
        nn.init.zeros_(self.final_projection.weight)
        nn.init.zeros_(self.final_projection.bias)
    
    def forward(
        self,
        visual_features: Tensor,
        text_embeddings: Tensor,
        style: Optional[Tensor] = None,
        text_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply text conditioning to visual features.
        
        Args:
            visual_features: Visual features (B, N_visual, feature_dim)
            text_embeddings: Text embeddings (B, N_text, text_dim)
            style: Style vector (B, style_dim) for modulation
            text_mask: Text attention mask (B, N_text)
            
        Returns:
            Text-conditioned visual features (B, N_visual, feature_dim)
        """
        # Project text embeddings to feature space
        text_features = self.text_projection(text_embeddings)  # (B, N_text, feature_dim)
        
        # Apply cross-attention layers
        x = visual_features
        
        for i in range(self.num_layers):
            # Cross-attention: visual features attend to text
            attn_output = self.cross_attention_layers[i](
                query=x,
                key=text_features,
                value=text_features,
                mask=text_mask,
            )
            
            # Residual connection
            x = x + attn_output
            
            # Normalization (adaptive or regular)
            if self.use_style_modulation and style is not None:
                x = self.norm_layers[i](x, style)
            else:
                x = self.norm_layers[i](x)
            
            # Feedforward
            ff_output = self.ff_layers[i](x)
            x = x + ff_output
        
        # Final projection with residual
        conditioned_features = visual_features + self.final_projection(x)
        
        return conditioned_features


class TextGuidedGenerator(nn.Module):
    """
    Wrapper that adds text conditioning capabilities to any generator.
    
    This module can be used to add text conditioning to existing generator
    architectures without modifying their core implementation.
    """
    
    def __init__(
        self,
        generator: nn.Module,
        text_encoder: CLIPTextEncoder,
        conditioning_module: TextConditioningModule,
        conditioning_layers: List[int] = None,
    ):
        """
        Initialize text-guided generator.
        
        Args:
            generator: Base generator model
            text_encoder: CLIP text encoder
            conditioning_module: Text conditioning module
            conditioning_layers: List of layer indices where conditioning is applied
        """
        super().__init__()
        
        self.generator = generator
        self.text_encoder = text_encoder
        self.conditioning_module = conditioning_module
        self.conditioning_layers = conditioning_layers or []
        
        # Register hooks for intermediate conditioning
        self._setup_conditioning_hooks()
    
    def _setup_conditioning_hooks(self):
        """Setup forward hooks for intermediate text conditioning."""
        self.conditioning_outputs = {}
        self.hooks = []
        
        # This is a simplified implementation
        # In practice, you'd need to identify specific layers in the generator
        # where text conditioning should be applied
        pass
    
    def forward(
        self,
        z: Tensor,
        texts: Optional[List[str]] = None,
        text_embeddings: Optional[Tensor] = None,
        **generator_kwargs,
    ) -> Dict[str, Tensor]:
        """
        Forward pass with text conditioning.
        
        Args:
            z: Input latent codes (B, latent_dim)
            texts: Text prompts (list of strings)
            text_embeddings: Pre-computed text embeddings (B, text_dim)
            **generator_kwargs: Additional generator arguments
            
        Returns:
            Generator outputs with text conditioning applied
        """
        # Get text embeddings
        if text_embeddings is None:
            if texts is not None:
                text_embeddings = self.text_encoder.encode_text(texts)
            else:
                # Use empty text embeddings
                batch_size = z.size(0)
                text_embeddings = torch.zeros(
                    batch_size, self.text_encoder.embedding_dim,
                    device=z.device, dtype=z.dtype
                )
        
        # Add text embeddings to generator kwargs
        generator_kwargs['text_embeddings'] = text_embeddings
        
        # Forward through generator
        return self.generator(z, **generator_kwargs)


def create_text_conditioning_system(
    feature_dim: int = 768,
    style_dim: int = 512,
    clip_model: str = "ViT-B/32",
    num_conditioning_layers: int = 2,
) -> Tuple[CLIPTextEncoder, TextConditioningModule]:
    """
    Factory function to create a complete text conditioning system.
    
    Args:
        feature_dim: Visual feature dimensionality
        style_dim: Style vector dimensionality
        clip_model: CLIP model variant
        num_conditioning_layers: Number of conditioning layers
        
    Returns:
        Tuple of (text_encoder, conditioning_module)
    """
    # Create text encoder
    text_encoder = CLIPTextEncoder(
        model_name=clip_model,
        freeze_weights=True,
    )
    
    # Create conditioning module
    conditioning_module = TextConditioningModule(
        feature_dim=feature_dim,
        text_dim=text_encoder.embedding_dim,
        num_layers=num_conditioning_layers,
        style_dim=style_dim,
        use_style_modulation=True,
    )
    
    return text_encoder, conditioning_module


def test_text_conditioning():
    """Test function for text conditioning system."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_size = 2
    feature_dim = 256
    style_dim = 128
    text_dim = 512
    
    print("Testing CLIP Text Encoder...")
    
    # Test text encoder
    text_encoder = CLIPTextEncoder().to(device)
    
    texts = ["a red car", "a blue house"]
    text_embeddings = text_encoder.encode_text(texts)
    
    print(f"Text embeddings shape: {text_embeddings.shape}")
    assert text_embeddings.shape[0] == len(texts)
    assert text_embeddings.shape[1] == text_encoder.embedding_dim
    
    print("Testing Text Conditioning Module...")
    
    # Test conditioning module
    conditioning_module = TextConditioningModule(
        feature_dim=feature_dim,
        text_dim=text_encoder.embedding_dim,
        style_dim=style_dim,
    ).to(device)
    
    # Generate test data
    seq_len = 64
    visual_features = torch.randn(batch_size, seq_len, feature_dim, device=device)
    style = torch.randn(batch_size, style_dim, device=device)
    
    # Apply conditioning
    conditioned_features = conditioning_module(
        visual_features, text_embeddings, style
    )
    
    print(f"Input features: {visual_features.shape}")
    print(f"Conditioned features: {conditioned_features.shape}")
    assert conditioned_features.shape == visual_features.shape
    
    print("Testing complete text conditioning system...")
    
    # Test factory function
    text_enc, cond_mod = create_text_conditioning_system(
        feature_dim=feature_dim,
        style_dim=style_dim,
    )
    
    print(f"Created text encoder with embedding dim: {text_enc.embedding_dim}")
    print(f"Created conditioning module with {cond_mod.num_layers} layers")
    
    print("All text conditioning tests passed!")


if __name__ == "__main__":
    test_text_conditioning()