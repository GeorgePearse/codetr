"""Main application for Co-DETR."""

import argparse
import torch
import logging

from codetr.configs import get_co_dino_inst_lvis_config
from codetr.models.co_dino_inst import build_co_dino_inst
from codetr.utils.load_weights import load_model_with_pretrained_weights


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


def init_weights(m):
    """Initialize model weights."""
    if isinstance(m, (torch.nn.Linear, torch.nn.Conv2d)):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.LayerNorm, torch.nn.GroupNorm)):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


def build_model(config, device):
    """Build Co-DETR model."""
    logger.info(f"Building {config.model_type} model...")
    
    # Create model based on type
    if config.model_type == 'co_dino_inst':
        model = build_co_dino_inst(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
        
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Load pretrained weights if available
    if config.pretrained and config.pretrained_weights:
        logger.info(f"Loading pre-trained weights from {config.pretrained_weights}...")
        model = load_model_with_pretrained_weights(
            model,
            model_name=config.pretrained_weights,
            strict=False,
            remap_keys=True,
        )
    else:
        logger.info("Initializing model with random weights")
        model.apply(init_weights)
        
    return model.to(device)


def test_forward_pass(model, config, device):
    """Test a forward pass with the model."""
    logger.info("Testing forward pass...")
    model.eval()
    
    # Create dummy input
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, config.img_size, config.img_size).to(device)
    
    with torch.no_grad():
        try:
            outputs = model(dummy_input)
            logger.info("Forward pass successful!")
            logger.info(f"Output keys: {outputs.keys()}")
            logger.info(f"Logits shape: {outputs['pred_logits'].shape}")
            logger.info(f"Boxes shape: {outputs['pred_boxes'].shape}")
            logger.info(f"Masks shape: {[m.shape for m in outputs['pred_masks']]}")
            return True
        except Exception as e:
            logger.error(f"Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    parser = argparse.ArgumentParser(description="Co-DETR main script")
    parser.add_argument('--model-type', type=str, default='co_dino_inst',
                        choices=['co_dino_inst'], help='Model type')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--pretrained', action='store_true', help='Use pretrained weights')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    args = parser.parse_args()
    
    # Get configuration
    if args.model_type == 'co_dino_inst':
        config = get_co_dino_inst_lvis_config()
    else:
        raise ValueError(f"Unsupported model type: {args.model_type}")
        
    # Override pretrained flag if specified
    if args.pretrained:
        config.pretrained = True
        
    # Build model
    model = build_model(config, args.device)
    
    # Test forward pass
    success = test_forward_pass(model, config, args.device)
    
    if success:
        logger.info("Model loaded and verified successfully!")
    else:
        logger.error("Model verification failed.")
    

if __name__ == "__main__":
    main()