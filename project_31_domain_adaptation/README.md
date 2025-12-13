# Project 31: Domain Adaptation

## Overview
Adapt RoBERTa to automotive domain.

## Model
- **Type**: RoBERTa-base
- **Parameters**: 125M
- **Difficulty**: ★★★★☆

## Techniques
- Continued pre-training
- Task-specific fine-tuning
- Domain vocabulary

## Performance
- Domain accuracy: +12%
- F1 Score: 92%

## Run
```bash
pip install transformers
python train.py
```
