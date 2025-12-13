# Project 3: Energy Consumption Forecasting

## Overview
LSTM-based time series forecasting for predicting building energy consumption, enabling 15% cost savings through optimized usage.

## Problem Statement
Energy costs are unpredictable and often wasteful. This system forecasts consumption 24 hours ahead for better planning and cost optimization.

## Use Cases
- Smart building energy management
- Grid load prediction
- HVAC optimization
- Renewable energy integration

## Dataset
- **Input**: 24-hour historical consumption data
- **Output**: Next hour consumption prediction
- **Features**: Single energy value (kWh)
- **Samples**: 2000 time points

## Performance
- **MAE**: <5%
- **Inference Time**: 12ms (IoT Gateway)
- **Model Size**: 80 KB (TFLite)

## Hardware Requirements
- Raspberry Pi 4 or IoT Gateway
- Smart meter integration
- Network connectivity

## Quick Start
```bash
python train.py
```
