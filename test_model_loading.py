"""Test script to verify model loading and weight compatibility."""

import argparse
from dataclasses import dataclass

import torch

from codetr.models.co_dino_inst import build_co_dino_inst
from codetr.utils.load_weights import load_model_with_pretrained_weights


@dataclass
class Config:
    """Model configuration."""
    # Model
    num_classes: int = 1203  # LVIS classes
    
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
    num_queries: int = 1500
    nheads: int = 8
    dim_feedforward: int = 2048
    enc_layers: int = 6
    dec_layers: int = 6
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
    mask_resolutions: list = None
    mask_use_semantic_branch: bool = True
    mask_use_instance_branch: bool = True
    mask_num_fusion_branches: int = 3
    mask_last_stage_agnostic: bool = True
    use_mask_iou: bool = True
    
    def __post_init__(self):
        if self.mask_resolutions is None:
            self.mask_resolutions = [14, 28, 56, 112]


def main():
    parser = argparse.ArgumentParser(description="Test Co-DINO-Inst model loading")
    parser.add_argument("--model-name", default="co_dino_inst_vit_l_lsj_lvis", 
                        help="Pre-trained model name")
    parser.add_argument("--strict", action="store_true", 
                        help="Use strict loading (fail on missing/unexpected keys)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    args = parser.parse_args()
    
    # Create config
    config = Config()
    
    # Build model
    print("Building Co-DINO-Inst model...")
    model = build_co_dino_inst(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load pre-trained weights
    print(f"\nLoading pre-trained weights from {args.model_name}...")
    model = load_model_with_pretrained_weights(
        model,
        model_name=args.model_name,
        strict=args.strict,
        remap_keys=True,
    )
    
    # Test forward pass
    print("\nTesting forward pass...")
    model = model.to(args.device)
    model.eval()
    
    # Create dummy input
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, config.img_size, config.img_size).to(args.device)
    
    with torch.no_grad():
        try:
            outputs = model(dummy_input)
            print("Forward pass successful!")
            print(f"Output keys: {outputs.keys()}")
            print(f"Logits shape: {outputs['pred_logits'].shape}")
            print(f"Boxes shape: {outputs['pred_boxes'].shape}")
            print(f"Masks shape: {[m.shape for m in outputs['pred_masks']]}")
        except Exception as e:
            print(f"Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nModel loading test completed!")


if __name__ == "__main__":
    main()