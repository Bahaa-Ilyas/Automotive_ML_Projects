# Architecture: Traffic Flow Prediction

## Model Architecture

```
Input (12, 4)
    ↓
GRU (64 units, return_sequences=True)
    ↓
GRU (32 units)
    ↓
Dense (16, ReLU)
    ↓
Output (4, Linear)
```

## Layer Details

| Layer | Type | Units | Activation | Output Shape | Parameters |
|-------|------|-------|------------|--------------|------------|
| Input | Input | - | - | (12, 4) | 0 |
| GRU1 | GRU | 64 | tanh | (12, 64) | 12,864 |
| GRU2 | GRU | 32 | tanh | (32,) | 9,312 |
| Dense1 | Dense | 16 | ReLU | (16,) | 528 |
| Output | Dense | 4 | Linear | (4,) | 68 |
| **Total** | | | | | **~22,772** |

## Data Flow

```
12h History (4 lanes) → MinMax Scaling → GRU → 1h Prediction → Inverse Scale
```

## Training Configuration
- **Optimizer**: Adam
- **Loss**: MSE
- **Metrics**: MAE
- **Epochs**: 30
- **Batch Size**: 32
- **Sequence Length**: 12 hours

## GRU vs LSTM
- **Faster**: 30% faster training/inference
- **Fewer Parameters**: Simpler gating mechanism
- **Similar Performance**: For this use case

## Deployment Pipeline

```
Keras Model → TFLite Converter → Quantization → Edge Server
```
