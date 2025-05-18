"""Utilities for loading pre-trained weights from HuggingFace."""

import os
from typing import Dict, Optional

import torch
from huggingface_hub import hf_hub_download


# Model configurations for Co-DINO-Inst
MODEL_CONFIGS = {
    "co_dino_5scale_lsj_resnet50_3x": {
        "repo_id": "zongzhuofan/Co-DINO",
        "filename": "co_dino_5scale_lsj_resnet50_3x.pth",
        "config": {
            "num_classes": 80,
            "hidden_dim": 256,
            "num_queries": 900,
            "num_co_heads": 2,
        }
    },
    "co_dino_5scale_lsj_swin_l_16e": {
        "repo_id": "zongzhuofan/Co-DINO",
        "filename": "co_dino_5scale_lsj_swin_l_16e.pth",
        "config": {
            "num_classes": 80,
            "hidden_dim": 256,
            "num_queries": 900,
            "num_co_heads": 2,
        }
    },
    "co_dino_inst_vit_l_lsj_lvis": {
        "repo_id": "zongzhuofan/Co-DINO",
        "filename": "co_dino_inst_vit_l_lsj_lvis.pth",
        "config": {
            "num_classes": 1203,  # LVIS classes
            "hidden_dim": 256,
            "num_queries": 1500,
            "num_co_heads": 2,
            "img_size": 1536,
            "window_size": 24,
            "num_feature_levels": 5,
            "enc_layers": 6,
            "dec_layers": 6,
            "mask_num_stages": 4,
            "mask_resolutions": [14, 28, 56, 112],
        }
    }
}


def download_pretrained_weights(
    model_name: str = "co_dino_inst_vit_l_lsj_lvis",
    cache_dir: Optional[str] = None,
) -> str:
    """Download pre-trained weights from HuggingFace.
    
    Args:
        model_name: Name of the model configuration
        cache_dir: Directory to cache the downloaded files
        
    Returns:
        Path to the downloaded checkpoint file
    """
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        
    config = MODEL_CONFIGS[model_name]
    
    # Download from HuggingFace Hub
    checkpoint_path = hf_hub_download(
        repo_id=config["repo_id"],
        filename=config["filename"],
        cache_dir=cache_dir,
    )
    
    return checkpoint_path


def load_pretrained_co_dino_inst(
    model_name: str = "co_dino_inst_vit_l_lsj_lvis",
    cache_dir: Optional[str] = None,
    map_location: str = "cpu",
) -> Dict:
    """Load pre-trained Co-DINO-Inst model weights.
    
    Args:
        model_name: Name of the model configuration
        cache_dir: Directory to cache the downloaded files
        map_location: Device to map the loaded tensors to
        
    Returns:
        Dictionary containing the model state dict and config
    """
    # Download weights
    checkpoint_path = download_pretrained_weights(model_name, cache_dir)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # Get model config
    config = MODEL_CONFIGS[model_name]["config"]
    
    return {
        "state_dict": checkpoint.get("model", checkpoint),
        "config": config,
        "checkpoint_path": checkpoint_path,
    }


def remap_checkpoint_keys(state_dict: Dict) -> Dict:
    """Remap checkpoint keys to match our implementation.
    
    Args:
        state_dict: Original state dict from checkpoint
        
    Returns:
        Remapped state dict
    """
    remapped = {}
    
    for key, value in state_dict.items():
        new_key = key
        
        # Remap backbone keys
        if key.startswith("backbone."):
            # Handle ViT backbone remapping
            if "patch_embed.proj" in key:
                new_key = key.replace("patch_embed.proj", "backbone.patch_embed")
            elif "blocks." in key:
                new_key = key.replace("blocks.", "backbone.blocks.")
            elif "norm." in key and "blocks." not in key:
                new_key = key.replace("norm.", "backbone.norm.")
                
        # Remap neck keys
        elif key.startswith("neck."):
            # SFP neck remapping
            if "lateral_convs" in key:
                new_key = key.replace("lateral_convs.0", "neck.lateral_conv")
            elif "fpn_convs" in key:
                # Map FPN convolutions to our SFP structure
                level_idx = int(key.split(".")[2])
                if level_idx == 0:  # P3
                    new_key = key.replace(f"fpn_convs.{level_idx}", "neck.p3_conv")
                elif level_idx == 1:  # P4
                    new_key = key.replace(f"fpn_convs.{level_idx}", "neck.p4_conv2")
                elif level_idx == 2:  # P5
                    new_key = key.replace(f"fpn_convs.{level_idx}", "neck.p5_conv2")
                    
        # Remap transformer keys
        elif key.startswith("transformer."):
            # Co-DINO transformer remapping
            new_key = key.replace("transformer.", "detection_head.transformer.")
            
        # Remap detection head keys
        elif key.startswith("bbox_head."):
            new_key = key.replace("bbox_head.", "detection_head.")
            
        # Remap mask head keys
        elif key.startswith("mask_head."):
            new_key = key.replace("mask_head.", "mask_head.")
            
        remapped[new_key] = value
        
    return remapped


def load_model_with_pretrained_weights(
    model: torch.nn.Module,
    model_name: str = "co_dino_inst_vit_l_lsj_lvis",
    strict: bool = False,
    remap_keys: bool = True,
) -> torch.nn.Module:
    """Load pre-trained weights into a Co-DINO-Inst model.
    
    Args:
        model: The Co-DINO-Inst model instance
        model_name: Name of the pre-trained model
        strict: Whether to strictly match all keys
        remap_keys: Whether to remap checkpoint keys
        
    Returns:
        Model with loaded weights
    """
    # Load pre-trained weights
    pretrained = load_pretrained_co_dino_inst(model_name)
    state_dict = pretrained["state_dict"]
    
    # Remap keys if needed
    if remap_keys:
        state_dict = remap_checkpoint_keys(state_dict)
        
    # Load into model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    
    print(f"Loaded pre-trained weights from {model_name}")
    if missing_keys:
        print(f"Missing keys: {len(missing_keys)}")
        print(f"First 10 missing keys: {missing_keys[:10]}")
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)}")
        print(f"First 10 unexpected keys: {unexpected_keys[:10]}")
        
    return model