# Project 38: Conversational RAG

## Overview
Multi-turn dialogue with Llama-2 7B.

## Models
- **Generator**: Llama-2-7B (7B)
- **Retriever**: Sentence-BERT (110M)
- **Difficulty**: ★★★★☆

## Performance
- Total: ~5-10 sec
- GPU: 10-20x faster

## Run
```bash
pip install sentence-transformers transformers
python train.py
```
