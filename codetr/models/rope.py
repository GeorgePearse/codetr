"""Rotary Position Embedding (RoPE) implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding for Vision Transformers."""
    
    def __init__(self, dim: int, max_freq: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_freq = max_freq
        
    def _get_freqs(self, h: int, w: int, device: torch.device) -> tuple:
        # Create frequency bands
        theta = self.max_freq ** (-torch.arange(0, self.dim, 2, device=device) / self.dim)
        
        # Create position indices
        y_pos = torch.arange(h, device=device).float()
        x_pos = torch.arange(w, device=device).float()
        
        # Create 2D position grid
        y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing='ij')
        
        # Apply frequencies to positions
        freqs_h = torch.outer(y_grid.flatten(), theta)
        freqs_w = torch.outer(x_grid.flatten(), theta)
        
        return freqs_h, freqs_w
        
    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Apply rotary position embedding.
        
        Args:
            x: Input tensor of shape (B, H, N, D) where H is heads, N is sequence length
            h: Height of the spatial grid
            w: Width of the spatial grid
            
        Returns:
            Tensor with rotary position embedding applied
        """
        B, H, N, D = x.shape
        assert N == h * w, f"Sequence length {N} must equal h*w ({h}*{w}={h*w})"
        
        # Get frequencies
        freqs_h, freqs_w = self._get_freqs(h, w, x.device)
        
        # Split features into two halves for sin/cos encoding
        x = rearrange(x, 'b h n d -> b h n (d_half two)', d_half=D//2, two=2)
        x_1, x_2 = x[..., 0], x[..., 1]
        
        # Apply rotary embedding
        cos_h = freqs_h.cos().view(1, 1, N, D//2)
        sin_h = freqs_h.sin().view(1, 1, N, D//2)
        cos_w = freqs_w.cos().view(1, 1, N, D//2)
        sin_w = freqs_w.sin().view(1, 1, N, D//2)
        
        # Rotate features
        x_rot_h = x_1 * cos_h - x_2 * sin_h
        x_rot_w = x_1 * cos_w + x_2 * sin_w
        
        # Combine rotated features
        x_rot = torch.stack([x_rot_h, x_rot_w], dim=-1)
        x_rot = rearrange(x_rot, 'b h n d_half two -> b h n (d_half two)')
        
        return x_rot