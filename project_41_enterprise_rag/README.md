# Project 41: Enterprise RAG

## Overview
Production-grade with Mixtral 8x7B MoE.

## Models
- **Generator**: Mixtral 8x7B (47B)
- **Retriever**: Sentence-BERT (110M)
- **Difficulty**: ★★★★★

## Performance
- Latency: <300ms
- Throughput: 100 queries/sec

## Run
```bash
pip install sentence-transformers faiss-cpu
python train.py
```
