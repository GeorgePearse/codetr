# codetr
Co-DETR implementation without MMDetection dependencies. This project provides a clean implementation of the Co-DINO-Inst model architecture.

## Overview
This implementation focuses on the ViT-L backbone variant of Co-DINO-Inst for instance segmentation, with the following components:
- Vision Transformer (ViT) Large backbone with patch size 16×16
- Simple Feature Pyramid (SFP) neck with 5 feature levels
- Co-DINO detection head with deformable transformer encoder and decoder
- Mask head with multi-stage refinement for instance segmentation

## Reference Implementation
Based on the official implementation at https://github.com/Sense-X/Co-DETR

## Target Model
Co-DINO-Inst with ViT-L backbone (LSJ, LVIS)

## Pre-trained Weights
Pre-trained weights are available at:
- LVIS model: https://huggingface.co/zongzhuofan (private repository)
- COCO instance segmentation model: https://huggingface.co/zongzhuofan/co-detr-vit-large-coco-instance

The code is configured to use the COCO instance segmentation model by default: `co_detr_vit_large_coco_instance`.

Note: Some repositories may require HuggingFace authentication.

## Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/codetr.git
cd codetr

# Install with uv
uv pip install -e .
```

## Usage

### Testing the Model
To test the model with a dummy forward pass:

```bash
python test_model_loading.py --device cpu
```

For loading with pre-trained weights, use:

```bash
python test_model_loading.py --device cuda --strict
```

### Main Application
Run the main application:

```bash
python main.py --model-type co_dino_inst --device cuda
```

## Model Architecture
The implementation includes the following key components:

1. **ViT-L Backbone**
   - Input resolution: 1536×1536 (configurable)
   - Patch size: 16×16
   - Embedding dimension: 1024
   - Depth: 24 layers
   - Window attention and rotary position encoding

2. **Simple Feature Pyramid (SFP) Neck**
   - Single-level to multi-level feature transformation
   - Output feature maps: P2-P6

3. **Co-DINO Detection Head**
   - Deformable transformer encoder (6 layers)
   - Deformable transformer decoder (6 layers)
   - Query denoising and collaborative training
   - Multi-scale attention

4. **Mask Head**
   - 4-stage mask refinement
   - Progressive upsampling: 14×14 → 28×28 → 56×56 → 112×112
   - Mask IoU prediction
