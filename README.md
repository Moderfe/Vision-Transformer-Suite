# Vision-Transformer-Suite

A comprehensive collection of state-of-the-art Vision Transformer (ViT) models implemented in PyTorch, optimized for both research and production.

## Models Included
- **ViT-Base/Large/Huge**: Standard transformer architectures for image classification.
- **DeiT**: Data-efficient Image Transformers.
- **Swin Transformer**: Hierarchical Vision Transformer using Shifted Windows.
- **MAE**: Masked Autoencoders for self-supervised learning.

## Key Features
- **Pre-trained Weights**: Easy access to ImageNet-1k and ImageNet-21k pre-trained models.
- **Flexible Training Pipeline**: Support for multi-GPU training via DistributedDataParallel (DDP).
- **Augmentation Library**: Integrated with advanced augmentations like RandAugment and Mixup.

## Quick Start
```python
from vit_suite import create_model

# Load a pre-trained Swin Transformer
model = create_model('swin_base_patch4_window7_224', pretrained=True)

# Inference
output = model(input_tensor)
```

## License
Apache License 2.0
