# Project 6: Traffic Flow Prediction

## Overview
GRU-based system for predicting traffic flow across multiple lanes, enabling smart city traffic management and congestion reduction.

## Problem Statement
Traffic congestion costs billions annually. This system predicts traffic flow 1 hour ahead for intelligent signal control and route optimization.

## Use Cases
- Smart traffic light control
- Route optimization
- Congestion prediction
- Emergency vehicle routing

## Dataset
- **Input**: 12-hour historical traffic (4 lanes)
- **Output**: Next hour prediction (4 lanes)
- **Samples**: 2000 time points

## Performance
- **MAE**: <8%
- **Inference Time**: 15ms (Edge Server)
- **Model Size**: 90 KB (TFLite)

## Hardware Requirements
- Edge server or Jetson Nano
- Traffic cameras/sensors
- Network connectivity

## Quick Start
```bash
python train.py
```
