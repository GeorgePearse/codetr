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
        
        # Apply rotary embedding directly without splitting
        # For simplicity, we'll apply a standard 2D RoPE
        pos_emb = torch.zeros_like(x)
        
        # Calculate position embeddings
        for i in range(0, D, 4):
            # Apply sin/cos to different dimensions
            if i < D:
                pos_emb[..., i] = freqs_h.cos().view(1, 1, N)
            if i + 1 < D:
                pos_emb[..., i + 1] = freqs_h.sin().view(1, 1, N)
            if i + 2 < D:
                pos_emb[..., i + 2] = freqs_w.cos().view(1, 1, N)
            if i + 3 < D:
                pos_emb[..., i + 3] = freqs_w.sin().view(1, 1, N)
                
        # Apply rotary position embedding
        x_rot = x * pos_emb
        
        return x_rot