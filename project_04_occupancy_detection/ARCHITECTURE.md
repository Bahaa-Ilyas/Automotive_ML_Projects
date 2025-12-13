# Architecture: Occupancy Detection

## Model Architecture

```
Input (5 features)
    ↓
Random Forest (50 trees, max_depth=10)
    ↓
Majority Vote
    ↓
Output (Binary: 0=Vacant, 1=Occupied)
```

## Feature Engineering

| Feature | Sensor | Range | Occupied Avg | Vacant Avg |
|---------|--------|-------|--------------|------------|
| Temperature | DHT22 | 18-30°C | 24°C | 21°C |
| Humidity | DHT22 | 20-70% | 50% | 38% |
| Light | LDR | 0-1000 lux | 500 lux | 150 lux |
| CO2 | MQ-135 | 400-1200 ppm | 800 ppm | 450 ppm |
| Sound | Mic | 0-100 dB | 40 dB | 15 dB |

## Model Details

| Parameter | Value |
|-----------|-------|
| Number of Trees | 50 |
| Max Depth | 10 |
| Min Samples Split | 2 |
| Features per Split | sqrt(5) ≈ 2 |
| Total Parameters | ~5000 nodes |

## Data Flow

```
Sensors → ADC → Feature Vector → Random Forest → Decision
```

## Training Configuration
- **Algorithm**: Random Forest
- **Criterion**: Gini impurity
- **Train/Test Split**: 80/20

## Deployment Pipeline

```
scikit-learn Model → Decision Tree Export → MicroPython Code → ESP32
```

## Edge Optimization
- Simplified to decision rules
- No floating point operations
- <2ms inference time
