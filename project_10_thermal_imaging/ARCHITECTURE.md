# Architecture: Thermal Imaging Analysis

## Model Architecture

```
Input (128, 128, 1)
    ↓
Conv2D (32 filters, 3x3, ReLU, same padding)
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (64 filters, 3x3, ReLU, same padding)
    ↓
MaxPooling2D (2x2)
    ↓
Conv2D (128 filters, 3x3, ReLU, same padding)
    ↓
GlobalAveragePooling2D
    ↓
Dense (64, ReLU)
    ↓
Output (3, Softmax)
```

## Layer Details

| Layer | Type | Filters/Units | Kernel | Output Shape | Parameters |
|-------|------|---------------|--------|--------------|------------|
| Input | Input | - | - | (128, 128, 1) | 0 |
| Conv1 | Conv2D | 32 | 3x3 | (128, 128, 32) | 320 |
| Pool1 | MaxPool2D | - | 2x2 | (64, 64, 32) | 0 |
| Conv2 | Conv2D | 64 | 3x3 | (64, 64, 64) | 18,496 |
| Pool2 | MaxPool2D | - | 2x2 | (32, 32, 64) | 0 |
| Conv3 | Conv2D | 128 | 3x3 | (32, 32, 128) | 73,856 |
| GAP | GlobalAvgPool | - | - | (128,) | 0 |
| Dense1 | Dense | 64 | - | (64,) | 8,256 |
| Output | Dense | 3 | - | (3,) | 195 |
| **Total** | | | | | **~101,123** |

## Thermal Image Processing

```
Thermal Camera → Grayscale (128x128) → Normalization → CNN → Classification
```

## Class Definitions

| Class | Description | Temperature Range |
|-------|-------------|-------------------|
| Normal | Expected thermal signature | Baseline ±5°C |
| Hot Spot | Excessive heat | >15°C above baseline |
| Cold Spot | Heat loss area | >10°C below baseline |

## Training Configuration
- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 25
- **Batch Size**: 16
- **Data Augmentation**: Rotation, flip

## Deployment Pipeline

```
Keras Model → TFLite Converter → GPU Delegate → Jetson Nano
```

## Edge Optimization
- GPU acceleration (CUDA)
- FP16 precision
- Batch size 1 for real-time
