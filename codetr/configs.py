"""Configuration classes for Co-DETR models."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class ModelConfig:
    """Base configuration for Co-DETR models."""
    # Model type
    model_type: str = 'co_dino_inst'
    
    # Basic
    num_classes: int = 80  # COCO
    pretrained: bool = False
    pretrained_weights: Optional[str] = None
    
    # Backbone
    img_size: int = 1536
    window_size: int = 24
    drop_path_rate: float = 0.4
    use_checkpoint: bool = True
    
    # Neck
    num_feature_levels: int = 5
    use_p2: bool = False
    
    # Transformer
    hidden_dim: int = 256
    num_queries: int = 900
    nheads: int = 8
    dim_feedforward: int = 2048
    enc_layers: int = 6
    dec_layers: int = 6
    pre_norm: bool = False
    dec_n_points: int = 4
    enc_n_points: int = 4
    num_patterns: int = 4
    num_co_heads: int = 2
    embed_init_tgt: bool = True
    
    # Detection head
    dn_number: int = 300
    dn_box_noise_scale: float = 0.4
    dn_label_noise_ratio: float = 0.5
    dn_batch_gt_fuse: bool = False
    
    # Mask head
    mask_num_stages: int = 4
    mask_channels: int = 256
    mask_resolutions: List[int] = field(default_factory=lambda: [14, 28, 56, 112])
    mask_use_semantic_branch: bool = True
    mask_use_instance_branch: bool = True
    mask_num_fusion_branches: int = 3
    mask_last_stage_agnostic: bool = True
    use_mask_iou: bool = True


@dataclass
class CoDINOInstConfig(ModelConfig):
    """Configuration for Co-DINO-Inst model."""
    model_type: str = 'co_dino_inst'
    
    # LVIS-specific settings
    num_classes: int = 1203  # LVIS
    num_queries: int = 1500


@dataclass
class TrainingConfig:
    """Training configuration for Co-DETR models."""
    # Basic
    seed: int = 42
    batch_size: int = 1
    num_workers: int = 4
    
    # Optimizer
    optimizer: str = 'adamw'
    lr: float = 5e-5
    weight_decay: float = 0.0001
    lr_backbone: float = 5e-6
    lr_backbone_names: List[str] = field(default_factory=lambda: ['backbone'])
    lr_linear_proj_names: List[str] = field(default_factory=lambda: ['reference_points', 'sampling_offsets'])
    lr_linear_proj_mult: float = 0.1
    layer_decay: float = 0.8
    
    # Schedule
    epochs: int = 12
    lr_drop: int = 11
    clip_max_norm: float = 0.1
    
    # Data augmentation
    use_lsj: bool = True
    min_size: int = 480
    max_size: int = 1536
    
    # EMA
    use_ema: bool = True
    ema_decay: float = 0.9997


def get_co_dino_inst_lvis_config() -> CoDINOInstConfig:
    """Get configuration for Co-DINO-Inst LVIS model."""
    return CoDINOInstConfig(
        model_type='co_dino_inst',
        num_classes=1203,
        pretrained=True,
        pretrained_weights='co_dino_inst_vit_l_lsj_lvis',
        img_size=1536,
        window_size=24,
        hidden_dim=256,
        num_queries=1500,
        num_co_heads=2,
    )