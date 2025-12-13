# Project 40: Agentic RAG

## Overview
Autonomous agent with tool use and planning.

## Models
- **Planner**: Llama-2-13B (13B)
- **Retriever**: Sentence-BERT (110M)
- **Difficulty**: ★★★★★

## Performance
- Total: ~5-8 sec
- Multi-step: 88%

## Run
```bash
pip install sentence-transformers
python train.py
```
