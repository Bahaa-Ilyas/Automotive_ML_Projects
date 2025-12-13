# Recommendation System Architecture

## Neural Collaborative Filtering

```
User ID → Embedding(50) ┐
                        ├→ Concat → Dense(128) → Dense(64) → Output(1)
Item ID → Embedding(50) ┘
```

## Components
1. **User Embedding**: 50-dimensional user representation
2. **Item Embedding**: 50-dimensional item representation
3. **Neural Network**: Learns interaction patterns
4. **Output**: Predicted rating (1-5 stars)

## Training
- Loss: MSE (Mean Squared Error)
- Optimizer: Adam
- Batch size: 64
- Epochs: 20-50

## Deployment
- REST API for real-time recommendations
- Batch processing for email campaigns
- Caching for popular items
