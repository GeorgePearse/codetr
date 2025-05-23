"""Co-DINO Transformer with deformable attention."""

import copy
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_, normal_

from codetr.models.deformable_attention import MSDeformableAttention
from codetr.models.position_encoding import build_position_embedding


class DeformableTransformerEncoderLayer(nn.Module):
    """Deformable transformer encoder layer."""
    
    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ):
        super().__init__()
        
        # Self attention
        self.self_attn = MSDeformableAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            num_levels=n_levels,
            num_points=n_points,
            dropout=dropout,
        )
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
        
    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src
        
    def forward(
        self,
        src: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        reference_points: Optional[torch.Tensor] = None,
        spatial_shapes: Optional[torch.Tensor] = None,
        level_start_index: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        # Self attention
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # FFN
        src = self.forward_ffn(src)
        
        return src


class DeformableTransformerEncoder(nn.Module):
    """Deformable transformer encoder."""
    
    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
                indexing='ij',
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
        
    def forward(
        self,
        src: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ):
        output = src
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src.device
        )
        
        for _, layer in enumerate(self.layers):
            output = layer(
                output,
                pos,
                reference_points,
                spatial_shapes,
                level_start_index,
                padding_mask,
            )
            
        return output


class DeformableTransformerDecoderLayer(nn.Module):
    """Deformable transformer decoder layer."""
    
    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
    ):
        super().__init__()
        
        # Self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross attention
        self.cross_attn = MSDeformableAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            num_levels=n_levels,
            num_points=n_points,
            dropout=dropout,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
        
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
        
    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt
        
    def forward(
        self,
        tgt: torch.Tensor,
        query_pos: Optional[torch.Tensor],
        reference_points: torch.Tensor,
        src: torch.Tensor,
        src_spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
    ):
        # Self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q.transpose(0, 1),
            k.transpose(0, 1),
            tgt.transpose(0, 1),
            attn_mask=self_attn_mask,
        )[0].transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross attention
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, query_pos),
            reference_points,
            src,
            src_spatial_shapes,
            level_start_index,
            src_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # FFN
        tgt = self.forward_ffn(tgt)
        
        return tgt


class DeformableTransformerDecoder(nn.Module):
    """Deformable transformer decoder."""
    
    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        return_intermediate: bool = False,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        
    def forward(
        self,
        tgt: torch.Tensor,
        reference_points: torch.Tensor,
        src: torch.Tensor,
        src_spatial_shapes: torch.Tensor,
        src_level_start_index: torch.Tensor,
        src_valid_ratios: torch.Tensor,
        query_pos: Optional[torch.Tensor] = None,
        src_padding_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
    ):
        output = tgt
        
        intermediate = []
        intermediate_reference_points = []
        
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
                )
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
                
            output = layer(
                output,
                query_pos,
                reference_points_input,
                src,
                src_spatial_shapes,
                src_level_start_index,
                src_padding_mask,
                self_attn_mask,
            )
            
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
                
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
            
        return output, reference_points


class CoDinoTransformer(nn.Module):
    """Co-DINO Transformer module."""
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        return_intermediate_dec: bool = True,
        num_feature_levels: int = 4,
        dec_n_points: int = 4,
        enc_n_points: int = 4,
        num_queries: int = 900,
        num_patterns: int = 4,
        num_co_heads: int = 2,
        embed_init_tgt: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_co_heads = num_co_heads
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.num_patterns = num_patterns
        self.num_encoder_layers = num_encoder_layers
        self.embed_init_tgt = embed_init_tgt
        
        # Encoder
        encoder_layer = DeformableTransformerEncoderLayer(
            d_model=d_model,
            d_ffn=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_levels=num_feature_levels,
            n_heads=nhead,
            n_points=enc_n_points,
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)
        
        # Decoder
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model=d_model,
            d_ffn=dim_feedforward,
            dropout=dropout,
            activation=activation,
            n_levels=num_feature_levels,
            n_heads=nhead,
            n_points=dec_n_points,
        )
        self.decoder = DeformableTransformerDecoder(
            decoder_layer, num_decoder_layers, return_intermediate=return_intermediate_dec
        )
        
        # Level embedding
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))
        
        # Query embedding
        self.query_embed = nn.Embedding(num_queries, d_model * 2)
        
        # Co-head patterns
        if num_patterns > 0:
            self.patterns = nn.Embedding(num_patterns, d_model)
            
        # Optional target embedding initialization
        if embed_init_tgt:
            self.tgt_embed = nn.Embedding(num_queries, d_model)
            
        self._reset_parameters()
        
    def _reset_parameters(self):
        xavier_uniform_(self.level_embed)
        normal_(self.query_embed.weight.data)
        
        if self.num_patterns > 0:
            normal_(self.patterns.weight.data)
            
        if self.embed_init_tgt:
            normal_(self.tgt_embed.weight.data)
            
    def gen_encoder_output_proposals(
        self,
        memory: torch.Tensor,
        memory_padding_mask: torch.Tensor,
        spatial_shapes: torch.Tensor,
    ):
        """Generate encoder output proposals for query selection."""
        N_, S_, C_ = memory.shape
        proposals = []
        _cur = 0
        
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
            
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
                indexing='ij',
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)
            
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            proposal = grid.view(N_, -1, 2)
            proposals.append(proposal)
            _cur += (H_ * W_)
            
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))
        
        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        
        return output_memory, output_proposals
        
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
        
    def forward(
        self,
        srcs: List[torch.Tensor],
        masks: List[torch.Tensor],
        pos_embeds: List[torch.Tensor],
        query_embed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of Co-DINO transformer.
        
        Args:
            srcs: List of multi-scale feature maps
            masks: List of masks for each feature map
            pos_embeds: List of positional embeddings
            query_embed: Optional query embeddings
            
        Returns:
            Dictionary containing decoder outputs and auxiliary outputs
        """
        assert self.num_feature_levels == len(srcs)
        
        # Prepare inputs
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
            
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        
        # Encoder
        memory = self.encoder(
            src_flatten,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            lvl_pos_embed_flatten,
            mask_flatten,
        )
        
        # Generate encoder output proposals
        output_memory, output_proposals = self.gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes
        )
        
        # Prepare queries
        query_embed, tgt = torch.split(query_embed, self.d_model, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        
        if self.embed_init_tgt:
            tgt = self.tgt_embed.weight.unsqueeze(0).expand(bs, -1, -1)
            
        # Initialize reference points
        reference_points = torch.sigmoid(output_proposals[:, :self.num_queries])
        init_reference_out = reference_points
        
        # Apply patterns for collaborative heads
        if self.num_patterns > 0:
            tgt_pat = self.patterns.weight.reshape(1, self.num_patterns, 1, self.d_model)
            tgt_pat = tgt_pat.repeat(bs, 1, self.num_queries // self.num_patterns, 1)
            tgt_pat = tgt_pat.reshape(bs, self.num_queries, self.d_model)
            tgt = tgt + tgt_pat
            
        # Decoder
        hs, inter_references = self.decoder(
            tgt,
            reference_points,
            memory,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            query_embed,
            mask_flatten,
        )
        
        inter_references_out = inter_references
        
        # Prepare outputs
        ret = {
            'hs': hs,  # (num_decoder_layers, B, num_queries, d_model)
            'init_reference': init_reference_out,  # (B, num_queries, 2/4)
            'inter_references': inter_references_out,  # (num_decoder_layers, B, num_queries, 2/4)
            'enc_outputs': output_memory,  # (B, num_enc_tokens, d_model)
            'enc_proposals': output_proposals,  # (B, num_enc_tokens, 2/4)
        }
        
        if self.num_co_heads > 1:
            # For collaborative training, duplicate outputs for each head
            ret['hs_co'] = hs.repeat(1, 1, self.num_co_heads, 1)
            ret['init_reference_co'] = init_reference_out.repeat(1, self.num_co_heads, 1)
            ret['inter_references_co'] = inter_references_out.repeat(1, 1, self.num_co_heads, 1)
            
        return ret


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def build_co_dino_transformer(args) -> CoDinoTransformer:
    """Build Co-DINO transformer from config."""
    return CoDinoTransformer(
        d_model=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation=args.activation,
        return_intermediate_dec=True,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        num_queries=args.num_queries,
        num_patterns=args.num_patterns,
        num_co_heads=args.num_co_heads,
        embed_init_tgt=args.embed_init_tgt,
    )