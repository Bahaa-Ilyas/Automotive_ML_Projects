"""=============================================================================
PROJECT 3: ENERGY CONSUMPTION FORECASTING - TRAINING SCRIPT
=============================================================================

PURPOSE:
This script trains an LSTM neural network to forecast energy consumption patterns
for smart buildings. By predicting future energy usage, buildings can optimize
HVAC systems, reduce costs by 20-30%, and participate in demand response programs.

WHY LSTM?
- Temporal Dependencies: Captures daily, weekly patterns in energy usage
- Long-term Memory: Remembers seasonal trends and historical patterns
- Sequential Data: Perfect for time-series forecasting
- Non-linear Patterns: Handles complex relationships in energy data

USE CASE:
Smart buildings use forecasts to:
- Pre-cool/heat buildings during off-peak hours (cheaper electricity)
- Participate in demand response programs (grid stability)
- Detect anomalies (equipment malfunction)
- Optimize renewable energy usage

INPUT DATA:
- Historical energy consumption (kWh)
- 24-hour lookback window
- Predicts next hour consumption
=============================================================================
"""

# STEP 1: Import Required Libraries
# -----------------------------------
import numpy as np  # Numerical operations
import tensorflow as tf  # Deep learning framework
from sklearn.preprocessing import MinMaxScaler  # Normalize data to 0-1 range
import sys
sys.path.append('..')  # Access shared utilities
from shared_utils.data_generator import SyntheticDataGenerator  # Generate synthetic data
from shared_utils.model_converter import ModelConverter  # Convert to TFLite

print("\n" + "="*70)
print("ENERGY CONSUMPTION FORECASTING - TRAINING")
print("="*70)

# STEP 2: Generate Synthetic Time Series Data
# --------------------------------------------
# In production, this would be real energy consumption data from smart meters
# Data represents hourly energy consumption in kWh
print("\n[1/7] Generating synthetic energy consumption data...")
data = SyntheticDataGenerator.time_series(n_samples=2000, n_features=1)
print(f"   ✓ Generated {len(data)} hourly readings")
print(f"   ✓ Time span: ~83 days of data")
print(f"   ✓ Sample values: {data[:5].flatten()}")

# STEP 3: Normalize the Data
# ---------------------------
# MinMaxScaler transforms data to range [0, 1]
# This helps LSTM learn faster and prevents gradient issues
# Formula: X_scaled = (X - X_min) / (X_max - X_min)
print("\n[2/7] Normalizing data to [0, 1] range...")
scaler = MinMaxScaler()
data = scaler.fit_transform(data)
print("   ✓ Data normalized")
print(f"   ✓ Min value: {data.min():.4f}")
print(f"   ✓ Max value: {data.max():.4f}")

# STEP 4: Create Sequences for Time Series Prediction
# ----------------------------------------------------
# Convert time series into supervised learning problem:
# Input: Last 24 hours of consumption
# Output: Next hour consumption
# Example: [hour1, hour2, ..., hour24] → hour25
print("\n[3/7] Creating sequences for time series prediction...")

def create_sequences(data, seq_length=24):
    """
    Create sequences for time series forecasting
    
    Args:
        data: Time series data
        seq_length: Number of past timesteps to use (24 hours)
    
    Returns:
        X: Input sequences (past 24 hours)
        y: Target values (next hour)
    
    Example:
        If seq_length=24:
        X[0] = [data[0], data[1], ..., data[23]]  # Past 24 hours
        y[0] = data[24]                            # Next hour to predict
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])  # Past 24 hours
        y.append(data[i+seq_length])     # Next hour
    return np.array(X), np.array(y)

X, y = create_sequences(data, seq_length=24)
print(f"   ✓ Created {len(X)} sequences")
print(f"   ✓ Input shape: {X.shape} (samples, timesteps, features)")
print(f"   ✓ Output shape: {y.shape}")
print(f"   ✓ Each input: 24 hours → predicts next hour")

# STEP 5: Split Data into Training and Testing Sets
# --------------------------------------------------
# 80% for training, 20% for testing
# Important: Use chronological split (not random) for time series
print("\n[4/7] Splitting data chronologically...")
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"   ✓ Training samples: {len(X_train)}")
print(f"   ✓ Testing samples: {len(X_test)}")
print("   ✓ Split is chronological (not random) for time series")

# STEP 6: Build the LSTM Neural Network
# --------------------------------------
# Architecture:
# - LSTM Layer 1: 64 units, returns sequences for next LSTM
# - LSTM Layer 2: 32 units, processes sequences
# - Dense Layer: 16 units with ReLU for feature extraction
# - Output Layer: 1 unit (predicted energy consumption)
print("\n[5/7] Building LSTM neural network...")
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(24, 1)),  # First LSTM
    tf.keras.layers.LSTM(32),  # Second LSTM
    tf.keras.layers.Dense(16, activation='relu'),  # Feature extraction
    tf.keras.layers.Dense(1)  # Output: predicted consumption
])

print("   ✓ Model architecture created")
print(f"   ✓ Total parameters: {model.count_params():,}")
print("   ✓ Input: 24 hours of consumption")
print("   ✓ Output: Next hour prediction")

# Compile the model
# - Adam optimizer: adaptive learning rate
# - MSE loss: Mean Squared Error (standard for regression)
# - MAE metric: Mean Absolute Error (easier to interpret)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
print("   ✓ Model compiled")
print("   ✓ Loss: MSE (Mean Squared Error)")
print("   ✓ Metric: MAE (Mean Absolute Error)")

# STEP 7: Train the Model
# -----------------------
# Train for 30 epochs with batch size of 32
# 20% of training data used for validation
print("\n[6/7] Training the model...")
print("   This may take 3-5 minutes...\n")
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# STEP 8: Evaluate the Model
# --------------------------
# Test on unseen data to measure forecasting accuracy
print("\n[7/7] Evaluating model performance...")
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"   ✓ Test MAE: {mae:.4f}")
print(f"   ✓ Test Loss (MSE): {loss:.4f}")
print(f"   ✓ Interpretation: Predictions are off by ~{mae:.4f} (normalized units)")

# STEP 9: Save the Models
# -----------------------
# Save in two formats:
# 1. Keras format (.h5) - full model
# 2. TFLite format - optimized for edge devices
print("\n[8/8] Saving models...")
model.save('energy_forecast_model.h5')
print("   ✓ Keras model saved: energy_forecast_model.h5")

ModelConverter.keras_to_tflite('energy_forecast_model.h5', 'model.tflite')
model_size = ModelConverter.get_model_size('model.tflite')
print(f"   ✓ TFLite model saved: model.tflite ({model_size:.2f} MB)")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\nModel Capabilities:")
print("  • Forecast energy consumption 1 hour ahead")
print("  • Identify daily and weekly patterns")
print("  • Detect anomalies in consumption")
print("  • Enable demand response optimization")
print("\nBusiness Impact:")
print("  • 20-30% reduction in energy costs")
print("  • Optimized HVAC scheduling")
print("  • Peak demand management")
print("  • Grid stability participation")
print("\nNext steps:")
print("1. Deploy to building management system")
print("2. Integrate with smart meter data")
print("3. Connect to HVAC control system")
print("4. Set up automated optimization")
print("="*70 + "\n")
