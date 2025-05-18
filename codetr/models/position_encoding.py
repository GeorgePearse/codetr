"""Positional encoding modules for Co-DETR."""

import math
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEmbeddingSine(nn.Module):
    """Sinusoidal positional encoding for 2D feature maps."""
    
    def __init__(
        self,
        num_feats: int = 128,
        temperature: int = 10000,
        normalize: bool = True,
        scale: float = 2 * math.pi,
        eps: float = 1e-6,
        offset: float = -0.5,
    ):
        super().__init__()
        self.num_feats = num_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = scale
        self.eps = eps
        self.offset = offset
        
    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """Generate sinusoidal positional encoding.
        
        Args:
            mask: Binary mask of shape (B, H, W) where True indicates padding
            
        Returns:
            Positional encoding of shape (B, C, H, W)
        """
        B, H, W = mask.shape
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            y_embed = y_embed / (y_embed[:, -1:, :] + self.eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + self.eps) * self.scale
            
        dim_t = torch.arange(self.num_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_feats)
        
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """Learned positional encoding."""
    
    def __init__(self, num_feats: int = 256, row_num_embed: int = 50, col_num_embed: int = 50):
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        
    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """Generate learned positional encoding.
        
        Args:
            mask: Binary mask of shape (B, H, W) where True indicates padding
            
        Returns:
            Positional encoding of shape (B, C, H, W)
        """
        B, H, W = mask.shape
        
        i = torch.arange(W, device=mask.device)
        j = torch.arange(H, device=mask.device)
        
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(H, 1, 1),
            y_emb.unsqueeze(1).repeat(1, W, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(B, 1, 1, 1)
        
        return pos


def build_position_embedding(
    hidden_dim: int,
    position_embedding_type: str = 'sine',
    temperature: int = 20,
) -> nn.Module:
    """Build positional encoding module.
    
    Args:
        hidden_dim: Hidden dimension of the model
        position_embedding_type: Type of positional encoding ('sine' or 'learned')
        temperature: Temperature for sine encoding
        
    Returns:
        Position embedding module
    """
    N_steps = hidden_dim // 2
    
    if position_embedding_type in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(
            num_feats=N_steps,
            temperature=temperature,
            normalize=True,
        )
    elif position_embedding_type in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(num_feats=N_steps)
    else:
        raise ValueError(f"not supported {position_embedding_type}")
        
    return position_embedding