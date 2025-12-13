# Architecture: Quality Control Vision

## Model Architecture

```
Input (224, 224, 3)
    ↓
MobileNetV2 Base (pretrained, frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dense (64, ReLU)
    ↓
Output (2, Softmax)
```

## MobileNetV2 Details

| Component | Description | Parameters |
|-----------|-------------|------------|
| Base Model | MobileNetV2 (ImageNet weights) | 2.3M |
| Inverted Residuals | 17 bottleneck blocks | - |
| Depthwise Separable | Efficient convolutions | - |
| Custom Head | 64-unit dense + output | 14K |
| **Total** | | **~2.3M** |

## Data Flow

```
Camera → Resize (224x224) → Normalize → MobileNetV2 → Classification
```

## Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 10
- **Batch Size**: 16
- **Data Augmentation**: Rotation, flip, brightness

## Deployment Pipeline

```
Keras Model → TFLite Converter → INT8 Quantization → Jetson Nano
```

## Edge Optimization
- **Quantization**: Full INT8 quantization
- **GPU Acceleration**: CUDA-enabled inference
- **Batch Size**: 1 (real-time)
