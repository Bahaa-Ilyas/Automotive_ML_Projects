# Project 36: Document QA RAG

## Overview
Extractive QA with DistilBERT.

## Models
- **Retriever**: Sentence-BERT (110M)
- **Generator**: DistilBERT (66M)
- **Difficulty**: ★★☆☆☆

## Performance
- Total latency: ~200ms
- No hallucinations

## Run
```bash
pip install transformers sentence-transformers
python train.py
```
