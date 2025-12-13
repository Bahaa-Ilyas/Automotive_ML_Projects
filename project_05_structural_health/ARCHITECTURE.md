# Architecture: Structural Health Monitoring

## Model Architecture

```
Input (10)
    ↓
Encoder:
  Dense (32, ReLU)
  Dense (16, ReLU)
  Dense (8, ReLU) [Latent Space]
    ↓
Decoder:
  Dense (16, ReLU)
  Dense (32, ReLU)
  Dense (10, Linear)
    ↓
Output (10) [Reconstruction]
```

## Layer Details

| Layer | Type | Units | Activation | Output Shape | Parameters |
|-------|------|-------|------------|--------------|------------|
| Input | Input | - | - | (10,) | 0 |
| Enc1 | Dense | 32 | ReLU | (32,) | 352 |
| Enc2 | Dense | 16 | ReLU | (16,) | 528 |
| Latent | Dense | 8 | ReLU | (8,) | 136 |
| Dec1 | Dense | 16 | ReLU | (16,) | 144 |
| Dec2 | Dense | 32 | ReLU | (32,) | 544 |
| Output | Dense | 10 | Linear | (10,) | 330 |
| **Total** | | | | | **~2,034** |

## Anomaly Detection

```
Input → Autoencoder → Reconstruction → MSE → Threshold → Anomaly Flag
```

**Threshold Calculation**: 95th percentile of training reconstruction errors

## Training Configuration
- **Optimizer**: Adam
- **Loss**: MSE (reconstruction error)
- **Epochs**: 50
- **Batch Size**: 32
- **Training Data**: Normal conditions only

## Deployment Pipeline

```
Keras Model → TFLite Converter → Quantization → Raspberry Pi
```

## Edge Optimization
- Lightweight architecture
- Fast inference (<20ms)
- Low memory footprint
