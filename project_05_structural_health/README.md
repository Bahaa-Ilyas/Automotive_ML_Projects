# Project 5: Structural Health Monitoring

## Overview
Autoencoder-based anomaly detection for monitoring bridge and building structural integrity, preventing catastrophic failures.

## Problem Statement
Structural failures are costly and dangerous. This system detects anomalies in vibration/strain patterns indicating potential structural issues.

## Use Cases
- Bridge health monitoring
- Building structural integrity
- Dam safety monitoring
- Wind turbine blade inspection

## Dataset
- **Input**: 10 sensor readings (strain gauges, accelerometers)
- **Output**: Anomaly score (reconstruction error)
- **Samples**: 1000 time points

## Performance
- **Detection Rate**: >90%
- **Inference Time**: 18ms (Raspberry Pi)
- **Model Size**: 100 KB (TFLite)

## Hardware Requirements
- Raspberry Pi 4
- Strain gauges (10x)
- Accelerometers
- Weatherproof enclosure

## Quick Start
```bash
python train.py
```
