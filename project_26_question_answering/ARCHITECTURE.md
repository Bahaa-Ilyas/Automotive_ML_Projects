# Architecture: Question Answering

## Model: DistilBERT
```
Question + Context → DistilBERT → Start/End Positions → Answer Span
```

## Components
- 6-layer transformer (distilled from BERT-12)
- 40% smaller, 60% faster
- 97% of BERT performance

## QA Process
1. Tokenize question + context
2. Predict start position
3. Predict end position
4. Extract answer span

## Next Steps
→ Project 27: Machine Translation with MarianMT (74M)
