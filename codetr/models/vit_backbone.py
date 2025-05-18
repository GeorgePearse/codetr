"""Vision Transformer (ViT) backbone for Co-DETR."""

from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import DropPath, Mlp
from timm.models.vision_transformer import Block

from codetr.models.rope import RotaryPositionEmbedding


class WindowAttention(nn.Module):
    """Window-based multi-head self attention (W-MSA) module with relative position bias."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        window_size: int = 0,
        use_rope: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.use_rope = use_rope
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        if use_rope:
            self.rope = RotaryPositionEmbedding(head_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # B, num_heads, N, head_dim
        
        if self.use_rope and self.window_size > 0:
            # Apply RoPE for window attention
            H = W = int(N ** 0.5)
            q = rearrange(q, 'b h (hw) c -> b h hw c', hw=H*W)
            k = rearrange(k, 'b h (hw) c -> b h hw c', hw=H*W)
            
            # Apply window attention if window_size is specified
            window_size = self.window_size
            q_windows = rearrange(q, 'b h (h_w w_w) (w_h w_w) c -> (b h_w w_w) h (w_h w_h) c',
                                w_h=window_size, w_w=window_size)
            k_windows = rearrange(k, 'b h (h_w w_w) (w_h w_w) c -> (b h_w w_w) h (w_h w_h) c',
                                w_h=window_size, w_w=window_size)
            v_windows = rearrange(v, 'b h (h_w w_w) (w_h w_w) c -> (b h_w w_w) h (w_h w_h) c',
                                w_h=window_size, w_w=window_size)
            
            # Apply RoPE to windowed q and k
            q_windows = self.rope(q_windows, window_size, window_size)
            k_windows = self.rope(k_windows, window_size, window_size)
            
            # Compute windowed attention
            attn = (q_windows @ k_windows.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x_windows = attn @ v_windows
            
            # Merge windows
            x = rearrange(x_windows, '(b h_w w_w) h (w_h w_h) c -> b h (h_w w_h) (w_w w_w) c',
                         h_w=H//window_size, w_w=W//window_size, w_h=window_size, w_w=window_size)
            x = rearrange(x, 'b h h_s w_s c -> b (h_s w_s) (h c)', h_s=H, w_s=W)
        else:
            # Standard attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class VitBlock(nn.Module):
    """Vision Transformer block with optional window attention."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        window_size: int = 0,
        use_rope: bool = False,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            window_size=window_size,
            use_rope=use_rope,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViTBackbone(nn.Module):
    """Vision Transformer backbone with window attention and RoPE support."""
    
    def __init__(
        self,
        img_size: int = 1536,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 2.667,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.4,
        norm_layer: nn.Module = nn.LayerNorm,
        act_layer: nn.Module = nn.GELU,
        use_abs_pos_embed: bool = False,
        use_rope: bool = True,
        window_size: int = 24,
        window_block_indexes: List[int] = None,
        use_checkpoint: bool = True,
        out_indices: List[int] = None,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.use_checkpoint = use_checkpoint
        self.out_indices = out_indices or [23]  # Default to last layer
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        num_patches = (img_size // patch_size) ** 2
        self.num_patches = num_patches
        
        # Position embedding
        if use_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        else:
            self.pos_embed = None
            
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # Stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # Window block indexes
        if window_block_indexes is None:
            # Default window attention pattern for ViT-L
            window_block_indexes = []
            for i in range(7):
                window_block_indexes.extend([i*4, i*4+1, i*4+2])
                
        # Transformer blocks
        self.blocks = nn.ModuleList([
            VitBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                window_size=window_size if i in window_block_indexes else 0,
                use_rope=use_rope,
            )
            for i in range(depth)
        ])
        
        self.norm = norm_layer(embed_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Patch embedding
        x = self.patch_embed(x)  # B, C, H, W
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        
        # Apply position embedding
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Pass through transformer blocks
        outputs = []
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
                
            if i in self.out_indices:
                # Reshape to spatial format and add to outputs
                B, N, C = x.shape
                H = W = int(N ** 0.5)
                out = x.transpose(1, 2).reshape(B, C, H, W)
                outputs.append(out)
        
        # Apply final norm if needed
        if len(outputs) == 0 or self.out_indices[-1] == len(self.blocks) - 1:
            x = self.norm(x)
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            out = x.transpose(1, 2).reshape(B, C, H, W)
            if len(outputs) == 0:
                outputs.append(out)
            else:
                outputs[-1] = out
                
        return outputs
        
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self.forward_features(x)


def build_vit_large_backbone(
    img_size: int = 1536,
    window_size: int = 24,
    drop_path_rate: float = 0.4,
    use_checkpoint: bool = True,
) -> ViTBackbone:
    """Build ViT-Large backbone for Co-DETR."""
    return ViTBackbone(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=2.667,
        qkv_bias=True,
        drop_path_rate=drop_path_rate,
        use_rope=True,
        window_size=window_size,
        window_block_indexes=None,  # Will use default pattern
        use_checkpoint=use_checkpoint,
    )