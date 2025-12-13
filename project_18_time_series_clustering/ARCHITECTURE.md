# Time Series Clustering Architecture

## Pipeline
```
Time Series → Normalize → Autoencoder → Embeddings → K-Means → Clusters
```

## Autoencoder Architecture
```
Encoder: Input(100) → Dense(64) → Dense(32) → Dense(16)
Decoder: Dense(16) → Dense(32) → Dense(64) → Output(100)
```

## Clustering
- Algorithm: K-Means
- Number of clusters: 3-10 (determined by elbow method)
- Distance metric: Euclidean in embedding space

## Training
1. Train autoencoder to reconstruct time series
2. Extract embeddings from encoder
3. Cluster embeddings with K-Means

## Deployment
- Batch processing for clustering large datasets
- Real-time classification of new time series
