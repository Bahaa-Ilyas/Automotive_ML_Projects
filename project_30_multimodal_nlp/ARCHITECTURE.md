# Architecture: Multimodal NLP

## Model: CLIP
```
Image → Vision Transformer → Embedding (512)
Text → Text Transformer → Embedding (512)
Similarity = Cosine(Image_Emb, Text_Emb)
```

## Training
- 400M image-text pairs
- Contrastive learning
- Zero-shot transfer

## Applications
- Visual search
- Image captioning
- Multimodal retrieval

## Next Steps
→ Project 31: Domain Adaptation with RoBERTa (125M)
