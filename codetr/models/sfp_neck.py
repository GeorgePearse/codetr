"""Simple Feature Pyramid (SFP) neck for Co-DETR."""

from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class SFPNeck(nn.Module):
    """Simple Feature Pyramid Network for ViT backbone.
    
    This neck creates a multi-scale feature pyramid from a single ViT output feature map.
    """
    
    def __init__(
        self,
        in_channels: int = 1024,
        out_channels: int = 256,
        num_outs: int = 5,
        start_level: int = 1,
        end_level: int = -1,
        add_extra_convs: bool = False,
        extra_convs_on_inputs: bool = True,
        relu_before_extra_convs: bool = False,
        no_norm_on_last: bool = True,
        conv_type: str = 'Conv2d',
        act_type: str = 'ReLU',
        norm_type: str = 'GroupNorm',
        upsample_type: str = 'bilinear',
        use_p2: bool = False,
    ):
        super().__init__()
        assert isinstance(in_channels, int)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = 1  # Single input from ViT
        self.num_outs = num_outs
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_last = no_norm_on_last
        self.use_p2 = use_p2
        
        if end_level == -1:
            self.num_outs = num_outs
        else:
            self.num_outs = end_level - start_level + 1
            
        # Build lateral convolutions
        self.lateral_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Build reduction convs for P3, P4, P5, P6
        self.reduction_convs = nn.ModuleList()
        self.reduction_norms = nn.ModuleList()
        
        # P2 (optional)
        if use_p2:
            # P2: 2x upsample -> Conv -> Norm -> GELU
            self.p2_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.p2_norm = nn.GroupNorm(32, out_channels)
            self.p2_act = nn.GELU()
        
        # P3: 2x upsample -> Conv -> Norm -> GELU
        self.p3_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.p3_norm = nn.GroupNorm(32, out_channels)
        self.p3_act = nn.GELU()
        
        # P4: 1x1 Conv -> Norm -> GELU -> 3x3 Conv -> Norm
        self.p4_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.p4_norm1 = nn.GroupNorm(32, out_channels)
        self.p4_act = nn.GELU()
        self.p4_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.p4_norm2 = nn.GroupNorm(32, out_channels)
        
        # P5: MaxPool -> 1x1 Conv -> Norm -> GELU -> 3x3 Conv -> Norm
        self.p5_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.p5_conv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.p5_norm1 = nn.GroupNorm(32, out_channels)
        self.p5_act = nn.GELU()
        self.p5_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.p5_norm2 = nn.GroupNorm(32, out_channels)
        
        # P6: Will be created by downsampling in the detection head
        if self.num_outs > 4:
            self.p6_maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.p6_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.p6_norm = nn.GroupNorm(32, out_channels)
            
        # Initialize weights
        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights of SFP."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass of SFP.
        
        Args:
            inputs: List containing a single feature map from ViT backbone
            
        Returns:
            List of multi-scale feature maps [P2, P3, P4, P5, P6]
        """
        assert len(inputs) == 1, "SFP expects single input from ViT"
        x = inputs[0]
        
        # Lateral connection
        lateral_out = self.lateral_conv(x)
        
        outs = []
        
        # P4: Direct mapping (same resolution)
        p4 = self.p4_conv1(lateral_out)
        p4 = self.p4_norm1(p4)
        p4 = self.p4_act(p4)
        p4 = self.p4_conv2(p4)
        p4 = self.p4_norm2(p4)
        
        # P3: 2x upsample
        p3 = F.interpolate(lateral_out, scale_factor=2, mode='bilinear', align_corners=False)
        p3 = self.p3_conv(p3)
        p3 = self.p3_norm(p3)
        p3 = self.p3_act(p3)
        
        # P2: 4x upsample (optional)
        if self.use_p2:
            p2 = F.interpolate(lateral_out, scale_factor=4, mode='bilinear', align_corners=False)
            p2 = self.p2_conv(p2)
            p2 = self.p2_norm(p2)
            p2 = self.p2_act(p2)
            outs.append(p2)
            
        outs.extend([p3, p4])
        
        # P5: 2x downsample
        p5 = self.p5_maxpool(lateral_out)
        p5 = self.p5_conv1(p5)
        p5 = self.p5_norm1(p5)
        p5 = self.p5_act(p5)
        p5 = self.p5_conv2(p5)
        p5 = self.p5_norm2(p5)
        outs.append(p5)
        
        # P6: 4x downsample (if needed)
        if self.num_outs > 4:
            p6 = self.p6_maxpool(p5)
            p6 = self.p6_conv(p6)
            p6 = self.p6_norm(p6)
            outs.append(p6)
            
        return outs