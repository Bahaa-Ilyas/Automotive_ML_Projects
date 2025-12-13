# Architecture: Energy Forecasting

## Model Architecture

```
Input (24, 1)
    ↓
LSTM (64 units, return_sequences=True)
    ↓
LSTM (32 units)
    ↓
Dense (16, ReLU)
    ↓
Output (1, Linear)
```

## Layer Details

| Layer | Type | Units | Activation | Output Shape | Parameters |
|-------|------|-------|------------|--------------|------------|
| Input | Input | - | - | (24, 1) | 0 |
| LSTM1 | LSTM | 64 | tanh | (24, 64) | 16,896 |
| LSTM2 | LSTM | 32 | tanh | (32,) | 12,416 |
| Dense1 | Dense | 16 | ReLU | (16,) | 528 |
| Output | Dense | 1 | Linear | (1,) | 17 |
| **Total** | | | | | **~29,857** |

## Data Flow

```
24h History → MinMax Scaling → Sequence → LSTM → Prediction → Inverse Scale
```

## Training Configuration
- **Optimizer**: Adam
- **Loss**: MSE
- **Metrics**: MAE
- **Epochs**: 30
- **Batch Size**: 32
- **Sequence Length**: 24 hours

## Deployment Pipeline

```
Keras Model → TFLite Converter → Quantization → IoT Gateway
```
