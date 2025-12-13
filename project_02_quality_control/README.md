# Project 2: Quality Control Vision System

## Overview
MobileNetV2-based computer vision system for automated defect detection in manufacturing, reducing manual inspection time by 80%.

## Problem Statement
Manual quality inspection is slow, inconsistent, and expensive. This system provides real-time automated defect detection with high accuracy.

## Use Cases
- PCB defect detection
- Surface scratch identification
- Product assembly verification
- Packaging quality control

## Dataset
- **Input**: RGB images (224x224x3)
- **Output**: Binary classification (OK/Defect)
- **Samples**: 500 synthetic images

## Performance
- **Accuracy**: ~92%
- **Inference Time**: 45ms (Jetson Nano)
- **Model Size**: 3.5 MB (TFLite)

## Hardware Requirements
- NVIDIA Jetson Nano
- USB/CSI camera (1080p)
- Lighting setup

## Quick Start
```bash
python train.py
python deploy_jetson.py
```
