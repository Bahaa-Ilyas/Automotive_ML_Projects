# Architecture: Hybrid RAG

## Pipeline
```
Query → Bi-encoder (fast) → Top-100 → Cross-encoder (accurate) → Top-10
```

## Trade-offs
- +150ms latency
- +17% precision

## Next Steps
→ Project 38: Add LLM with Llama-2 7B
