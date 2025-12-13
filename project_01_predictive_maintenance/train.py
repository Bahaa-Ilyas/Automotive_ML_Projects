"""=============================================================================
PROJECT 1: PREDICTIVE MAINTENANCE - TRAINING SCRIPT
=============================================================================

PURPOSE:
This script trains an LSTM neural network to predict equipment failures before
they occur. By analyzing sensor data (vibration, temperature, pressure), the
model learns patterns that indicate an impending failure.

WHY LSTM?
LSTM (Long Short-Term Memory) networks are perfect for time-series data because
they can remember patterns over time, which is crucial for detecting gradual
equipment degradation.

OUTPUT:
- Trained model file (.h5)
- Optimized TFLite model for edge deployment
- Performance metrics (accuracy)
=============================================================================
"""

# STEP 1: Import Required Libraries
# -----------------------------------
import numpy as np  # For numerical operations and array handling
import tensorflow as tf  # Deep learning framework for building neural networks
from sklearn.model_selection import train_test_split  # Split data into train/test sets
from sklearn.preprocessing import StandardScaler  # Normalize data to improve training
import sys
sys.path.append('..')  # Add parent directory to path to access shared utilities
from shared_utils.data_generator import SyntheticDataGenerator  # Generate synthetic sensor data
from shared_utils.model_converter import ModelConverter  # Convert models to edge-friendly formats

print("\n" + "="*70)
print("PREDICTIVE MAINTENANCE MODEL TRAINING")
print("="*70)

# STEP 2: Generate Synthetic Sensor Data
# ---------------------------------------
# In production, this would be real sensor data from equipment
# We generate 5000 samples with 5% anomaly rate (equipment failures)
print("\n[1/6] Generating synthetic sensor data...")
X, y = SyntheticDataGenerator.sensor_data(n_samples=5000)
print(f"   ✓ Generated {len(X)} samples")
print(f"   ✓ Normal samples: {np.sum(y == 0)}")
print(f"   ✓ Failure samples: {np.sum(y == 1)}")

# STEP 3: Split Data into Training and Testing Sets
# --------------------------------------------------
# 80% for training, 20% for testing
# This ensures we can evaluate the model on unseen data
print("\n[2/6] Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   ✓ Training samples: {len(X_train)}")
print(f"   ✓ Testing samples: {len(X_test)}")

# STEP 4: Normalize the Data
# --------------------------
# StandardScaler transforms data to have mean=0 and std=1
# This helps the neural network learn faster and more effectively
print("\n[3/6] Normalizing data...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit on training data and transform
X_test = scaler.transform(X_test)  # Only transform test data (no fitting)
print("   ✓ Data normalized (mean=0, std=1)")

# STEP 5: Build the LSTM Neural Network
# --------------------------------------
# Architecture:
# - Input: Single sensor value at each time step
# - LSTM Layer 1: 32 units, returns sequences for next LSTM
# - LSTM Layer 2: 16 units, processes sequences
# - Dense Layer: 8 units with ReLU activation for feature extraction
# - Output Layer: 1 unit with sigmoid (outputs probability 0-1)
print("\n[4/6] Building LSTM neural network...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1, 1)),  # Input shape: (timesteps, features)
    tf.keras.layers.LSTM(32, return_sequences=True),  # First LSTM layer
    tf.keras.layers.LSTM(16),  # Second LSTM layer
    tf.keras.layers.Dense(8, activation='relu'),  # Hidden layer
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output: probability of failure
])

# Compile the model with optimizer and loss function
# - Adam optimizer: adaptive learning rate
# - Binary crossentropy: loss function for binary classification
# - Accuracy: metric to monitor during training
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("   ✓ Model architecture created")
print(f"   ✓ Total parameters: {model.count_params():,}")

# Reshape data for LSTM input: (samples, timesteps, features)
X_train_seq = X_train.reshape(-1, 1, 1)
X_test_seq = X_test.reshape(-1, 1, 1)

# STEP 6: Train the Model
# -----------------------
# Train for 20 epochs with batch size of 32
# Validation split: 20% of training data used for validation during training
print("\n[5/6] Training the model...")
print("   This may take 2-5 minutes...\n")
history = model.fit(
    X_train_seq, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# STEP 7: Evaluate the Model
# --------------------------
# Test the model on unseen data to measure real-world performance
print("\n[6/6] Evaluating model performance...")
loss, acc = model.evaluate(X_test_seq, y_test, verbose=0)
print(f"   ✓ Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"   ✓ Test Loss: {loss:.4f}")

# STEP 8: Save the Models
# -----------------------
# Save in two formats:
# 1. Keras format (.h5) - for further training or analysis
# 2. TFLite format - optimized for edge devices (Raspberry Pi)
print("\n[7/7] Saving models...")
model.save('predictive_maintenance_model.h5')
print("   ✓ Keras model saved: predictive_maintenance_model.h5")

ModelConverter.keras_to_tflite('predictive_maintenance_model.h5', 'model.tflite')
model_size = ModelConverter.get_model_size('model.tflite')
print(f"   ✓ TFLite model saved: model.tflite ({model_size:.2f} MB)")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\nNext steps:")
print("1. Deploy model.tflite to Raspberry Pi")
print("2. Run deploy_raspberry_pi.py for real-time predictions")
print("3. Connect actual sensors for production use")
print("="*70 + "\n")
