# Architecture: Document Classification

## Model Architecture

```
Input (100)
    ↓
Embedding (5000 vocab → 64 dim)
    ↓
GlobalAveragePooling1D
    ↓
Dense (32, ReLU)
    ↓
Output (5, Softmax)
```

## Layer Details

| Layer | Type | Params | Output Shape | Description |
|-------|------|--------|--------------|-------------|
| Input | Input | - | (100,) | Token IDs |
| Embedding | Embedding | 320K | (100, 64) | Word vectors |
| Pooling | GlobalAvgPool | 0 | (64,) | Sequence aggregation |
| Dense1 | Dense | 2,080 | (32,) | Feature extraction |
| Output | Dense | 165 | (5,) | Classification |
| **Total** | | **~322K** | | |

## Text Processing Pipeline

```
Raw Text → Tokenization → Padding/Truncation → Embedding → Classification
```

## Training Configuration
- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 20
- **Batch Size**: 32
- **Max Sequence Length**: 100

## Embedding Strategy
- **Trainable**: Yes
- **Dimension**: 64 (compact for edge)
- **Vocabulary**: 5000 most common words

## Deployment Pipeline

```
Keras Model → TFLite Converter → Quantization → Mobile/Edge Device
```

## Edge Optimization
- Small vocabulary (5K vs 50K+)
- Compact embeddings (64 vs 300)
- Global pooling (vs LSTM/GRU)
