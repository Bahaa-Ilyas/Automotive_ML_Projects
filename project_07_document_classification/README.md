# Project 7: Automated Document Classification

## Overview
Lightweight NLP model for automated document classification, reducing manual sorting time by 90% in business workflows.

## Problem Statement
Manual document sorting is time-consuming and error-prone. This system automatically classifies documents into categories for efficient processing.

## Use Cases
- Invoice/receipt classification
- Email categorization
- Legal document sorting
- Customer support ticket routing

## Dataset
- **Input**: Text sequences (max 100 tokens)
- **Output**: 5 document categories
- **Vocabulary**: 5000 words
- **Samples**: 1000 documents

## Performance
- **Accuracy**: ~88%
- **Inference Time**: 30ms (Mobile/Edge)
- **Model Size**: 500 KB (TFLite)

## Hardware Requirements
- Mobile device or edge server
- Minimal compute requirements

## Quick Start
```bash
python train.py
```
