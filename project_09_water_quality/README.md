# Project 9: Water Quality Monitoring

## Overview
Gradient Boosting classifier for real-time water quality assessment using IoT sensors, ensuring safe drinking water and environmental compliance.

## Problem Statement
Water quality testing is slow and expensive. This system provides real-time quality assessment for immediate action on contamination.

## Use Cases
- Drinking water safety monitoring
- Industrial wastewater compliance
- River/lake environmental monitoring
- Aquaculture water management

## Dataset
- **Input**: 5 water parameters (pH, turbidity, DO, conductivity, temp)
- **Output**: Binary (Good/Poor quality)
- **Samples**: 1500 readings

## Performance
- **Accuracy**: ~96%
- **Inference Time**: <5ms (LoRa device)
- **Model Size**: 15 KB (converted)

## Hardware Requirements
- LoRa module (SX1276)
- pH sensor
- Turbidity sensor
- Dissolved oxygen sensor
- Conductivity sensor
- Temperature sensor

## Quick Start
```bash
python train.py
python deploy_lora.py
```
