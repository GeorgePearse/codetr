"""Co-DINO head for object detection."""

from typing import List, Dict, Optional, Tuple
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, normal_

from codetr.models.co_dino_transformer import CoDinoTransformer, build_co_dino_transformer


class MLP(nn.Module):
    """Simple multi-layer perceptron (also called FFN)."""
    
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class CoDINOHead(nn.Module):
    """Co-DINO detection head with query denoising."""
    
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 256,
        num_queries: int = 900,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        enc_layers: int = 6,
        dec_layers: int = 6,
        num_patterns: int = 4,
        num_co_heads: int = 2,
        pre_norm: bool = False,
        num_feature_levels: int = 5,
        dec_n_points: int = 4,
        enc_n_points: int = 4,
        dn_number: int = 300,
        dn_box_noise_scale: float = 0.4,
        dn_label_noise_ratio: float = 0.5,
        dn_batch_gt_fuse: bool = False,
        embed_init_tgt: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_co_heads = num_co_heads
        self.num_feature_levels = num_feature_levels
        self.dn_number = dn_number
        self.dn_box_noise_scale = dn_box_noise_scale
        self.dn_label_noise_ratio = dn_label_noise_ratio
        self.dn_batch_gt_fuse = dn_batch_gt_fuse
        
        # Build transformer
        self.transformer = CoDinoTransformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="relu",
            return_intermediate_dec=True,
            num_feature_levels=num_feature_levels,
            dec_n_points=dec_n_points,
            enc_n_points=enc_n_points,
            num_queries=num_queries,
            num_patterns=num_patterns,
            num_co_heads=num_co_heads,
            embed_init_tgt=embed_init_tgt,
        )
        
        # Classification heads
        self.class_embed = nn.ModuleList()
        for _ in range(num_co_heads):
            self.class_embed.append(
                nn.Linear(hidden_dim, num_classes)
            )
            
        # Bounding box regression heads
        self.bbox_embed = nn.ModuleList()
        for _ in range(num_co_heads):
            bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
            self.bbox_embed.append(bbox_embed)
            
        # Multi-layer heads
        self.num_pred_layers = dec_layers
        self.class_embed = nn.ModuleList(
            [copy.deepcopy(self.class_embed) for _ in range(self.num_pred_layers)]
        )
        self.bbox_embed = nn.ModuleList(
            [copy.deepcopy(self.bbox_embed) for _ in range(self.num_pred_layers)]
        )
        
        # Encoder classification layer
        self.enc_output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        
        # Encoder bbox output
        self.enc_bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.enc_class_embed = nn.Linear(hidden_dim, num_classes)
        
        # For P6 feature level
        if num_feature_levels > 4:
            self.extra_level_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, 2, 1)
            
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        # Initialize classification heads
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        for class_embed in self.class_embed:
            for head in class_embed:
                head.bias.data = torch.ones(self.num_classes) * bias_value
                
        # Initialize encoder classification
        self.enc_class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        
        # Initialize bbox regression heads
        for bbox_layers in self.bbox_embed:
            for bbox_embed in bbox_layers:
                constant_(bbox_embed.layers[-1].weight.data, 0)
                constant_(bbox_embed.layers[-1].bias.data, 0)
                
        constant_(self.enc_bbox_embed.layers[-1].weight.data, 0)
        constant_(self.enc_bbox_embed.layers[-1].bias.data, 0)
        
    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord, outputs_class):
        """Set auxiliary losses for intermediate predictions."""
        return [
            {'pred_logits': a, 'pred_boxes': b}
            for a, b in zip(outputs_class[:-1], outputs_coord[:-1])
        ]
        
    def prepare_for_dn(self, targets):
        """Prepare denoising targets for training."""
        if self.training and self.dn_number > 0:
            # This is a simplified version - full implementation would include
            # query denoising preparation with noisy GT boxes and labels
            return None, None, None, 0
        return None, None, None, 0
        
    def forward(self, srcs, masks, pos_embeds, targets=None):
        """Forward pass of Co-DINO head.
        
        Args:
            srcs: List of multi-scale feature maps
            masks: List of masks for each feature map
            pos_embeds: List of positional embeddings
            targets: Optional targets for training (with denoising)
            
        Returns:
            Dictionary containing predictions
        """
        assert len(srcs) >= 4, "Co-DINO requires at least 4 feature levels"
        
        # Add extra P6 level if needed
        if self.num_feature_levels > 4 and len(srcs) == 4:
            p6 = self.extra_level_conv(srcs[-1])
            srcs.append(p6)
            mask = F.interpolate(masks[-1][None].float(), size=p6.shape[-2:]).to(torch.bool)[0]
            masks.append(mask)
            # Duplicate last pos_embed for P6
            pos_embeds.append(pos_embeds[-1])
            
        # Query denoising preparation (training only)
        dn_meta = None
        if self.training:
            dn_query_embed, dn_attn_mask, dn_meta, dn_query_num = self.prepare_for_dn(targets)
        else:
            dn_query_embed, dn_attn_mask, dn_query_num = None, None, 0
            
        # Get query embeddings
        query_embed = self.transformer.query_embed.weight
        
        # Merge DN queries if available
        if dn_query_embed is not None:
            query_embed = torch.cat([dn_query_embed, query_embed], dim=0)
            
        # Forward through transformer
        hs = self.transformer(srcs, masks, pos_embeds, query_embed)
        
        # Split denoising and matching queries
        if dn_query_num > 0:
            dn_hs = hs['hs'][:, :, :dn_query_num, :]
            hs['hs'] = hs['hs'][:, :, dn_query_num:, :]
            
        # Get predictions from each decoder layer
        outputs_coords = []
        outputs_classes = []
        
        for layer_id in range(self.num_pred_layers):
            # Process predictions for each collaborative head
            layer_coords = []
            layer_classes = []
            
            for head_id in range(self.num_co_heads):
                # Extract features for this head
                if self.num_co_heads > 1:
                    hs_head = hs['hs'][layer_id, :, head_id::self.num_co_heads, :]
                else:
                    hs_head = hs['hs'][layer_id]
                    
                # Classification
                outputs_class = self.class_embed[layer_id][head_id](hs_head)
                layer_classes.append(outputs_class)
                
                # Box regression
                tmp = self.bbox_embed[layer_id][head_id](hs_head)
                
                # Add reference points
                if 'init_reference' in hs:
                    if self.num_co_heads > 1:
                        reference = hs['init_reference'][:, head_id::self.num_co_heads]
                    else:
                        reference = hs['init_reference']
                    if reference.shape[-1] == 4:
                        tmp = tmp + reference
                    else:
                        assert reference.shape[-1] == 2
                        tmp[..., :2] = tmp[..., :2] + reference
                        
                outputs_coord = tmp.sigmoid()
                layer_coords.append(outputs_coord)
                
            outputs_coords.append(torch.stack(layer_coords, dim=1))
            outputs_classes.append(torch.stack(layer_classes, dim=1))
            
        # Stack outputs: [num_layers, batch_size, num_co_heads, num_queries, ...]
        outputs_coord = torch.stack(outputs_coords)
        outputs_class = torch.stack(outputs_classes)
        
        # Get encoder outputs for mixed query selection
        enc_outputs = self.enc_output(hs['enc_outputs'])
        
        # Encoder class and bbox predictions
        enc_outputs_class = self.enc_class_embed(enc_outputs)
        enc_outputs_coord = self.enc_bbox_embed(enc_outputs).sigmoid()
        
        # Prepare final outputs
        out = {
            'pred_logits': outputs_class[-1],  # Final layer predictions
            'pred_boxes': outputs_coord[-1],
            'aux_outputs': self._set_aux_loss(outputs_coord, outputs_class),
        }
        
        # Add encoder outputs for mixed query selection
        out['enc_outputs'] = {
            'pred_logits': enc_outputs_class,
            'pred_boxes': enc_outputs_coord,
        }
        
        # Add DN outputs if in training
        if dn_meta is not None:
            # Process DN predictions similarly
            # This is simplified - full implementation would include proper DN loss computation
            pass
            
        return out, dn_meta


def build_co_dino_head(args) -> CoDINOHead:
    """Build Co-DINO head from config."""
    return CoDINOHead(
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        nheads=args.nheads,
        dim_feedforward=args.dim_feedforward,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        num_patterns=args.num_patterns,
        num_co_heads=args.num_co_heads,
        pre_norm=args.pre_norm,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        dn_number=args.dn_number,
        dn_box_noise_scale=args.dn_box_noise_scale,
        dn_label_noise_ratio=args.dn_label_noise_ratio,
        dn_batch_gt_fuse=args.dn_batch_gt_fuse,
        embed_init_tgt=args.embed_init_tgt,
    )