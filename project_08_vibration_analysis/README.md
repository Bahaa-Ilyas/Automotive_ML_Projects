# Project 8: Machinery Vibration Analysis

## Overview
1D CNN for real-time machinery fault detection through vibration analysis, preventing equipment failures and reducing maintenance costs.

## Problem Statement
Machinery faults cause unexpected downtime. This system detects bearing faults, misalignment, and imbalance through vibration signature analysis.

## Use Cases
- Bearing fault detection
- Motor condition monitoring
- Pump vibration analysis
- Gearbox health monitoring

## Dataset
- **Input**: 128-sample vibration signals
- **Output**: Binary (Normal/Fault)
- **Frequency**: 50 Hz (normal), 120 Hz (fault)
- **Samples**: 400 signals

## Performance
- **Accuracy**: ~93%
- **Inference Time**: 12ms (ESP32 with TFLite Micro)
- **Model Size**: 60 KB (TFLite)

## Hardware Requirements
- ESP32 or Arduino Nano 33 BLE
- ADXL345 accelerometer
- Power supply

## Quick Start
```bash
python train.py
```
