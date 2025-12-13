# Architecture: Semantic Search RAG

## Pipeline
```
Documents → Sentence-BERT → Embeddings → FAISS Index
Query → Sentence-BERT → Search → Top-K Docs
```

## Advantages
- Semantic understanding
- Handles paraphrases
- Better recall

## Next Steps
→ Project 36: Add QA with DistilBERT
