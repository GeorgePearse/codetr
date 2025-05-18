"""Instance segmentation mask head for Co-DINO-Inst."""

from typing import List, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBlock(nn.Module):
    """Basic convolutional block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        norm_layer: Optional[nn.Module] = None,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.norm = norm_layer if norm_layer is not None else nn.Identity()
        self.activation = activation if activation is not None else nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class MultiBranchFusionAvg(nn.Module):
    """Multi-branch feature fusion using average."""
    
    def __init__(self, in_channels: int, out_channels: int, num_branches: int = 3):
        super().__init__()
        self.num_branches = num_branches
        self.branches = nn.ModuleList([
            ConvBlock(
                in_channels,
                out_channels,
                kernel_size=3,
                dilation=rate,
                padding=rate,
                norm_layer=nn.GroupNorm(32, out_channels),
                activation=nn.ReLU(inplace=True),
            )
            for rate in [1, 3, 5][:num_branches]
        ])
        
        self.fusion = ConvBlock(
            out_channels * num_branches,
            out_channels,
            kernel_size=1,
            padding=0,
            norm_layer=nn.GroupNorm(32, out_channels),
            activation=nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        features = [branch(x) for branch in self.branches]
        x = torch.cat(features, dim=1)
        x = self.fusion(x)
        return x


class SimpleRefineMaskHead(nn.Module):
    """Simple refine mask head for instance segmentation with multi-stage refinement."""
    
    def __init__(
        self,
        num_stages: int = 4,
        hidden_dim: int = 256,
        mask_channels: int = 256,
        num_classes: int = 80,
        mask_resolutions: List[int] = [14, 28, 56, 112],
        use_semantic_branch: bool = True,
        use_instance_branch: bool = True,
        num_fusion_branches: int = 3,
        last_stage_agnostic: bool = True,
    ):
        super().__init__()
        self.num_stages = num_stages
        self.hidden_dim = hidden_dim
        self.mask_channels = mask_channels
        self.num_classes = num_classes
        self.mask_resolutions = mask_resolutions
        self.use_semantic_branch = use_semantic_branch
        self.use_instance_branch = use_instance_branch
        self.last_stage_agnostic = last_stage_agnostic
        
        # Stage-wise refinement modules
        self.stage_convs = nn.ModuleList()
        self.stage_upsamples = nn.ModuleList()
        self.stage_fusion = nn.ModuleList()
        self.stage_outputs = nn.ModuleList()
        
        for stage in range(num_stages):
            # Current resolution
            curr_res = mask_resolutions[stage]
            prev_res = mask_resolutions[stage - 1] if stage > 0 else curr_res // 2
            
            # Input channels for this stage
            in_channels = hidden_dim if stage == 0 else mask_channels
            
            # Instance feature branch
            if use_instance_branch:
                instance_conv = nn.Sequential(
                    ConvBlock(
                        in_channels,
                        mask_channels,
                        kernel_size=3,
                        padding=1,
                        norm_layer=nn.GroupNorm(32, mask_channels),
                        activation=nn.ReLU(inplace=True),
                    ),
                    ConvBlock(
                        mask_channels,
                        mask_channels,
                        kernel_size=3,
                        padding=1,
                        norm_layer=nn.GroupNorm(32, mask_channels),
                        activation=nn.ReLU(inplace=True),
                    ),
                )
            else:
                instance_conv = nn.Identity()
                
            # Semantic feature branch
            if use_semantic_branch:
                semantic_conv = nn.Sequential(
                    ConvBlock(
                        in_channels,
                        mask_channels,
                        kernel_size=3,
                        padding=1,
                        norm_layer=nn.GroupNorm(32, mask_channels),
                        activation=nn.ReLU(inplace=True),
                    ),
                    ConvBlock(
                        mask_channels,
                        mask_channels,
                        kernel_size=3,
                        padding=1,
                        norm_layer=nn.GroupNorm(32, mask_channels),
                        activation=nn.ReLU(inplace=True),
                    ),
                )
            else:
                semantic_conv = nn.Identity()
                
            self.stage_convs.append(nn.ModuleDict({
                'instance': instance_conv,
                'semantic': semantic_conv,
            }))
            
            # Feature fusion
            fusion_in_channels = 0
            if use_instance_branch:
                fusion_in_channels += mask_channels
            if use_semantic_branch:
                fusion_in_channels += mask_channels
            if stage > 0:
                fusion_in_channels += mask_channels  # Previous stage output
                
            self.stage_fusion.append(
                MultiBranchFusionAvg(
                    fusion_in_channels,
                    mask_channels,
                    num_branches=num_fusion_branches,
                )
            )
            
            # Upsampling
            if curr_res > prev_res:
                upsample = nn.ConvTranspose2d(
                    mask_channels,
                    mask_channels,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                )
            else:
                upsample = nn.Identity()
            self.stage_upsamples.append(upsample)
            
            # Stage output
            # Last stage is class-agnostic
            out_channels = 1 if (last_stage_agnostic and stage == num_stages - 1) else num_classes
            self.stage_outputs.append(
                nn.Conv2d(mask_channels, out_channels, kernel_size=1)
            )
            
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(
        self,
        features: torch.Tensor,
        instance_features: Optional[torch.Tensor] = None,
        semantic_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, List[torch.Tensor]]:
        """Forward pass of mask head.
        
        Args:
            features: Query features from decoder of shape (B, N, C)
            instance_features: Optional instance-level features
            semantic_features: Optional semantic-level features from FPN
            
        Returns:
            Dictionary containing mask predictions at each stage
        """
        B, N, C = features.shape
        
        # Reshape to spatial format for first stage
        h = w = self.mask_resolutions[0]
        features_2d = features.view(B * N, C, 1, 1)
        features_2d = F.interpolate(features_2d, size=(h, w), mode='bilinear', align_corners=False)
        
        outputs = []
        prev_output = None
        
        for stage in range(self.num_stages):
            # Get features for this stage
            if stage == 0:
                stage_features = features_2d
            else:
                stage_features = prev_output
                
            # Process through branches
            branch_features = []
            
            if self.use_instance_branch:
                instance_feat = self.stage_convs[stage]['instance'](stage_features)
                branch_features.append(instance_feat)
                
            if self.use_semantic_branch:
                semantic_feat = self.stage_convs[stage]['semantic'](stage_features)
                branch_features.append(semantic_feat)
                
            # Add previous stage output if available
            if stage > 0 and prev_output is not None:
                branch_features.append(prev_output)
                
            # Fuse features
            if len(branch_features) > 1:
                fused_features = torch.cat(branch_features, dim=1)
            else:
                fused_features = branch_features[0]
                
            fused_features = self.stage_fusion[stage](fused_features)
            
            # Generate mask prediction
            mask_pred = self.stage_outputs[stage](fused_features)
            
            # Reshape back to (B, N, num_classes/1, H, W)
            output_shape = mask_pred.shape
            mask_pred = mask_pred.view(B, N, output_shape[1], output_shape[2], output_shape[3])
            outputs.append(mask_pred)
            
            # Upsample for next stage
            if stage < self.num_stages - 1:
                prev_output = self.stage_upsamples[stage](fused_features)
            
        return {'masks': outputs}


class MaskIoUHead(nn.Module):
    """Mask IoU prediction head for mask quality assessment."""
    
    def __init__(
        self,
        hidden_dim: int = 256,
        mask_dim: int = 256,
        num_convs: int = 2,
        num_fc: int = 1,
    ):
        super().__init__()
        self.num_convs = num_convs
        self.num_fc = num_fc
        
        # Convolutional layers
        self.convs = nn.ModuleList()
        for i in range(num_convs):
            in_channels = mask_dim if i == 0 else hidden_dim
            self.convs.append(
                ConvBlock(
                    in_channels,
                    hidden_dim,
                    kernel_size=3,
                    padding=1,
                    norm_layer=nn.GroupNorm(32, hidden_dim),
                    activation=nn.ReLU(inplace=True),
                )
            )
            
        # Fully connected layers
        self.fcs = nn.ModuleList()
        for i in range(num_fc):
            in_features = hidden_dim if i == 0 else hidden_dim
            out_features = hidden_dim if i < num_fc - 1 else 1
            self.fcs.append(nn.Linear(in_features, out_features))
            
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, mask_features: torch.Tensor, mask_pred: torch.Tensor) -> torch.Tensor:
        """Forward pass of MaskIoU head.
        
        Args:
            mask_features: Mask features of shape (B, C, H, W)
            mask_pred: Mask predictions of shape (B, 1, H, W)
            
        Returns:
            IoU predictions of shape (B, 1)
        """
        # Concatenate mask features and predictions
        x = torch.cat([mask_features, mask_pred], dim=1)
        
        # Convolutional layers
        for conv in self.convs:
            x = conv(x)
            
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.flatten(1)
        
        # Fully connected layers
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < len(self.fcs) - 1:
                x = F.relu(x)
                
        return x.sigmoid()


def build_mask_head(args) -> SimpleRefineMaskHead:
    """Build mask head from config."""
    return SimpleRefineMaskHead(
        num_stages=args.mask_num_stages,
        hidden_dim=args.hidden_dim,
        mask_channels=args.mask_channels,
        num_classes=args.num_classes,
        mask_resolutions=args.mask_resolutions,
        use_semantic_branch=args.mask_use_semantic_branch,
        use_instance_branch=args.mask_use_instance_branch,
        num_fusion_branches=args.mask_num_fusion_branches,
        last_stage_agnostic=args.mask_last_stage_agnostic,
    )