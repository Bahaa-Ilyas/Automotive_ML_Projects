# Architecture: Basic RAG

## Pipeline
```
Query → TF-IDF → Cosine Similarity → Top-K Docs → Template → Response
```

## Components
- TF-IDF vectorization
- Cosine similarity
- Template generation

## Limitations
- No semantic understanding
- Keyword-dependent

## Next Steps
→ Project 35: Add neural embeddings
