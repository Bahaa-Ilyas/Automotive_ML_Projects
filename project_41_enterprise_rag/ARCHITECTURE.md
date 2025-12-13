# Architecture: Enterprise RAG

## Pipeline
```
Query → Hybrid Retrieval → FAISS → Re-ranking → Mixtral 8x7B → Cache
```

## MoE Architecture
- 8 experts (47B total)
- 2 active per token (13B)
- 6x faster than dense

## Enterprise Features
- Caching, Audit, Citations
- Access Control, Monitoring

## Next Steps
→ Project 42: Specialize for code with CodeLlama 34B
