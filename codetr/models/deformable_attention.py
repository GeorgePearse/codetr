"""Multi-scale deformable attention module."""

import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_


def _is_power_of_2(n):
    if (n & (n - 1)) == 0 and n != 0:
        return True
    return False


class MSDeformableAttention(nn.Module):
    """Multi-scale deformable attention module."""
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}")
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        
        self.sampling_offsets = nn.Linear(embed_dim, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dim, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self._reset_parameters()
        
    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2
        ).repeat(1, self.num_levels, self.num_points, 1)
        
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
            
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
            
        constant_(self.attention_weights.weight.data, 0.0)
        constant_(self.attention_weights.bias.data, 0.0)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.0)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.0)
        
    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        input_flatten: torch.Tensor,
        input_spatial_shapes: torch.Tensor,
        input_level_start_index: torch.Tensor,
        input_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass of multi-scale deformable attention.
        
        Args:
            query: Query features of shape (B, L_q, C)
            reference_points: Reference points of shape (B, L_q, n_levels, 2) in range [0, 1]
            input_flatten: Flattened input features of shape (B, L_in, C)
            input_spatial_shapes: Spatial shapes of each level (n_levels, 2)
            input_level_start_index: Start index of each level in input_flatten
            input_padding_mask: Padding mask of shape (B, L_in)
            
        Returns:
            Output features of shape (B, L_q, C)
        """
        B, L_q, _ = query.shape
        B, L_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == L_in
        
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(B, L_in, self.num_heads, self.embed_dim // self.num_heads)
        
        sampling_offsets = self.sampling_offsets(query).view(
            B, L_q, self.num_heads, self.num_levels, self.num_points, 2
        )
        attention_weights = self.attention_weights(query).view(
            B, L_q, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = F.softmax(attention_weights, -1).view(
            B, L_q, self.num_heads, self.num_levels, self.num_points
        )
        
        # B, L_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
            )
        else:
            raise ValueError(f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}")
            
        # Use grid_sample for bilinear interpolation
        output = self._sample_locations(
            value, sampling_locations, attention_weights, input_spatial_shapes,
            input_level_start_index, input_padding_mask
        )
        output = self.output_proj(output)
        
        return self.dropout(output)
        
    def _sample_locations(
        self,
        value: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
        input_spatial_shapes: torch.Tensor,
        input_level_start_index: torch.Tensor,
        input_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample values at given locations using bilinear interpolation."""
        B, _, n_heads, C_v = value.shape
        _, L_q, _, n_levels, n_points, _ = sampling_locations.shape
        
        value_list = value.split(
            [H_ * W_ for H_, W_ in input_spatial_shapes.tolist()], dim=1
        )
        sampling_grids = 2 * sampling_locations - 1
        
        output = torch.zeros(
            B, L_q, n_heads, C_v, dtype=value.dtype, device=value.device
        )
        
        for level_id, (H_, W_) in enumerate(input_spatial_shapes.tolist()):
            value_l = value_list[level_id].view(B, H_, W_, n_heads, C_v)
            value_l = value_l.permute(0, 3, 4, 1, 2)  # B, n_heads, C_v, H_, W_
            
            sampling_grid_l = sampling_grids[:, :, :, level_id, :, :].view(
                B, L_q, n_heads, n_points, 2
            )
            sampling_grid_l = sampling_grid_l.view(B, L_q * n_heads, n_points, 2)
            
            for i in range(n_points):
                sampling_grid_li = sampling_grid_l[:, :, i, :].view(B, L_q, n_heads, 1, 2)
                sampling_grid_li = sampling_grid_li.expand(-1, -1, -1, C_v, -1)
                
                sampled_value_l = F.grid_sample(
                    value_l.view(B * n_heads, C_v, H_, W_),
                    sampling_grid_li.view(B * n_heads, L_q, C_v, 2),
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=False,
                )
                
                sampled_value_l = sampled_value_l.view(B, n_heads, C_v, L_q).permute(0, 3, 1, 2)
                
                attention_weight_l = attention_weights[:, :, :, level_id, i].view(
                    B, L_q, n_heads, 1
                )
                
                output += sampled_value_l * attention_weight_l
                
        output = output.view(B, L_q, n_heads * C_v)
        
        return output