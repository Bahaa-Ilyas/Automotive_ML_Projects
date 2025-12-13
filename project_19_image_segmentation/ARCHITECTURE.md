# Image Segmentation Architecture

## U-Net Architecture
```
Input(256×256×3)
    ↓
Encoder: Conv → Pool → Conv → Pool
    ↓
Bottleneck: Conv(256)
    ↓
Decoder: UpSample + Skip → Conv → UpSample + Skip → Conv
    ↓
Output(256×256×3) - Pixel-level classification
```

## Key Features
1. **Skip Connections**: Preserve spatial information
2. **Encoder-Decoder**: Downsample then upsample
3. **Pixel-level Output**: Each pixel classified

## Training
- Loss: Sparse Categorical Crossentropy
- Metric: Pixel accuracy, IoU
- Batch size: 8
- Epochs: 25-50

## Deployment
- Medical imaging: Tumor segmentation
- Autonomous vehicles: Road/obstacle segmentation
- Satellite imagery: Land use classification
