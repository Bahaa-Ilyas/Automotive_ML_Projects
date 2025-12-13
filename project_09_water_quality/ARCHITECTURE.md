# Architecture: Water Quality Monitoring

## Model Architecture

```
Input (5 features)
    ↓
StandardScaler (normalization)
    ↓
Gradient Boosting (100 trees, depth=5)
    ↓
Ensemble Prediction
    ↓
Output (Binary: 0=Poor, 1=Good)
```

## Feature Details

| Feature | Sensor | Range | Good Quality | Poor Quality |
|---------|--------|-------|--------------|--------------|
| pH | pH probe | 0-14 | 6.5-8.5 | <6 or >9 |
| Turbidity | Nephelometer | 0-100 NTU | <5 NTU | >10 NTU |
| Dissolved O2 | DO probe | 0-20 mg/L | >6 mg/L | <4 mg/L |
| Conductivity | EC probe | 0-2000 μS/cm | 200-800 | >1000 |
| Temperature | Thermistor | 0-40°C | 15-25°C | >30°C |

## Model Details

| Parameter | Value |
|-----------|-------|
| Algorithm | Gradient Boosting |
| Number of Trees | 100 |
| Max Depth | 5 |
| Learning Rate | 0.1 |
| Subsample | 0.8 |
| Total Parameters | ~500 nodes |

## Data Flow

```
Sensors → ADC → Feature Vector → Scaling → Gradient Boosting → Quality Score
```

## Training Configuration
- **Algorithm**: Gradient Boosting Classifier
- **Loss**: Log loss
- **Train/Test Split**: 80/20
- **Cross-validation**: 5-fold

## Deployment Pipeline

```
scikit-learn Model → Simplified Rules → LoRa Device
```

## Edge Optimization
- Threshold-based rules for edge
- Minimal computation
- Battery-efficient (<5ms)
- LoRaWAN for long-range transmission
