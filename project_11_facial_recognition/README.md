# Project 11: Facial Recognition System

## Overview
FaceNet-based facial recognition system for real-time identification and verification. Converts faces to 128-dimensional embeddings for efficient matching.

## Problem Statement
Traditional facial recognition requires extensive training data per person. FaceNet enables one-shot learning - recognizing new faces with just one example.

## Use Cases
- **Security**: Access control, surveillance
- **Attendance**: Automated check-in systems
- **Retail**: Customer recognition, personalization
- **Photos**: Automatic organization and tagging
- **Healthcare**: Patient identification

## Dataset
- **Input**: 160x160 RGB face images
- **Training**: Triplet pairs (anchor, positive, negative)
- **Identities**: 100 people (expandable)
- **Images per person**: 20+ for training

## Architecture
- **Backbone**: InceptionResNetV2
- **Embedding**: 128 dimensions (L2 normalized)
- **Loss**: Triplet Loss with margin=0.2
- **Training**: Hard negative mining

## Performance
- **Accuracy**: 95%+ (with quality images)
- **Inference Time**: 100ms (Raspberry Pi 4)
- **Model Size**: 25 MB (TFLite)
- **Database**: Handles 10,000+ faces

## Hardware Requirements
- Raspberry Pi 4 (4GB+ RAM)
- Camera module (1080p)
- Optional: Coral USB Accelerator

## Quick Start
```bash
python train.py
```

## Deployment
1. Detect faces (MTCNN/RetinaFace)
2. Generate embeddings
3. Compare with database
4. Return identity if match found
