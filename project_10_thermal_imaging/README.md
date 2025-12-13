# Project 10: Thermal Imaging Analysis

## Overview
CNN-based thermal imaging analysis for building inspection and heat loss detection, reducing energy waste by 20% through targeted improvements.

## Problem Statement
Energy audits are expensive and time-consuming. This system automatically identifies heat loss areas and equipment hotspots from thermal images.

## Use Cases
- Building energy audits
- Electrical panel inspection
- HVAC system diagnostics
- Solar panel fault detection

## Dataset
- **Input**: Thermal images (128x128x1)
- **Output**: 3 classes (Normal/Hot Spot/Cold Spot)
- **Samples**: 400 thermal images

## Performance
- **Accuracy**: ~89%
- **Inference Time**: 35ms (Jetson Nano)
- **Model Size**: 2 MB (TFLite)

## Hardware Requirements
- NVIDIA Jetson Nano
- FLIR Lepton or similar thermal camera
- Mounting hardware

## Quick Start
```bash
python train.py
python deploy_jetson.py
```
