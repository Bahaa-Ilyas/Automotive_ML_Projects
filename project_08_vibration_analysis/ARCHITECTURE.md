# Architecture: Vibration Analysis

## Model Architecture

```
Input (128, 1)
    ↓
Conv1D (32 filters, kernel=3, ReLU)
    ↓
MaxPooling1D (pool_size=2)
    ↓
Conv1D (64 filters, kernel=3, ReLU)
    ↓
MaxPooling1D (pool_size=2)
    ↓
Flatten
    ↓
Dense (32, ReLU)
    ↓
Output (1, Sigmoid)
```

## Layer Details

| Layer | Type | Filters/Units | Kernel | Output Shape | Parameters |
|-------|------|---------------|--------|--------------|------------|
| Input | Input | - | - | (128, 1) | 0 |
| Conv1 | Conv1D | 32 | 3 | (126, 32) | 128 |
| Pool1 | MaxPool1D | - | 2 | (63, 32) | 0 |
| Conv2 | Conv1D | 64 | 3 | (61, 64) | 6,208 |
| Pool2 | MaxPool1D | - | 2 | (30, 64) | 0 |
| Flatten | Flatten | - | - | (1920,) | 0 |
| Dense1 | Dense | 32 | - | (32,) | 61,472 |
| Output | Dense | 1 | - | (1,) | 33 |
| **Total** | | | | | **~67,841** |

## Signal Processing

```
Accelerometer → ADC → 128 samples → Normalization → 1D CNN → Fault Detection
```

## Training Configuration
- **Optimizer**: Adam
- **Loss**: Binary Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 30
- **Batch Size**: 16

## 1D CNN Advantages
- **Temporal patterns**: Captures frequency signatures
- **Translation invariant**: Detects faults at any position
- **Efficient**: Fewer parameters than 2D CNN

## Deployment Pipeline

```
Keras Model → TFLite Converter → TFLite Micro → ESP32/Arduino
```

## Edge Optimization
- INT8 quantization
- TFLite Micro runtime
- <100KB total footprint
