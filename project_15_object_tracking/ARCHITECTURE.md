# Object Tracking Architecture

## Siamese Network

```
Template (127x127) → CNN → Features(256) ┐
                                          ├→ Similarity → Output(1)
Search (255x255) → CNN → Features(256)   ┘
```

## Components
1. **Feature Extractor**: Shared CNN for both branches
2. **Similarity Matching**: Cosine similarity or correlation
3. **Output**: Probability that object is in search region

## Tracking Pipeline
1. Initialize with bounding box in first frame
2. Extract template features
3. For each subsequent frame:
   - Extract search region features
   - Compute similarity map
   - Find maximum similarity location
   - Update bounding box

## Real-time Optimization
- Lightweight CNN backbone
- TensorRT optimization
- Frame skipping for high-speed tracking
