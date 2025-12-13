# Project 42: Code RAG

## Overview
Specialized RAG for code with CodeLlama 34B.

## Models
- **Generator**: CodeLlama-34B (34B)
- **Retriever**: Sentence-BERT (110M)
- **Difficulty**: ★★★★★

## Performance
- Generation: ~30 tokens/sec
- Context: 100K tokens

## Run
```bash
pip install sentence-transformers
python train.py
```
