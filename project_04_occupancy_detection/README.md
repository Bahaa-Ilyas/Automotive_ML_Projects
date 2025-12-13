# Project 4: Smart Building Occupancy Detection

## Overview
Random Forest classifier for real-time room occupancy detection using environmental sensors, reducing HVAC costs by 30%.

## Problem Statement
Buildings waste energy heating/cooling empty rooms. This system detects occupancy in real-time for intelligent climate control.

## Use Cases
- Smart HVAC control
- Meeting room management
- Energy optimization
- Security monitoring

## Dataset
- **Input**: 5 sensor readings (temp, humidity, light, CO2, sound)
- **Output**: Binary classification (Occupied/Vacant)
- **Samples**: 2000 sensor readings

## Performance
- **Accuracy**: ~94%
- **Inference Time**: 2ms (ESP32)
- **Model Size**: 20 KB (converted)

## Hardware Requirements
- ESP32 DevKit
- DHT22 (temp/humidity)
- Light sensor (LDR)
- MQ-135 (CO2)
- Sound sensor

## Quick Start
```bash
python train.py
python deploy_esp32.py  # Upload to ESP32
```
