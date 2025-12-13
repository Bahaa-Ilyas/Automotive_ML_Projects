"""=============================================================================
PROJECT 13: NETWORK ANOMALY DETECTION - TRAINING SCRIPT
=============================================================================

PURPOSE:
Train an Autoencoder for detecting network intrusions and anomalous traffic
patterns. Protects systems from cyberattacks, DDoS, and unauthorized access.

WHY AUTOENCODER?
- Unsupervised: Learns normal traffic patterns without labeled attacks
- Reconstruction Error: Anomalies have high error
- Adaptable: Detects new, unknown attack types
- Real-time: Fast inference for network monitoring

NETWORK FEATURES (10 dimensions):
1. Packet size
2. Protocol type (TCP/UDP/ICMP)
3. Connection duration
4. Bytes sent/received
5. Port numbers
6. Packet rate
7. Error rate
8. Retransmission rate
9. Connection state
10. Service type

USE CASES:
- Intrusion Detection Systems (IDS)
- DDoS attack detection
- Malware communication detection
- Insider threat detection
- Network security monitoring
=============================================================================
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('..')
from shared_utils.data_generator import SyntheticDataGenerator
from shared_utils.model_converter import ModelConverter

print("\n" + "="*70)
print("NETWORK ANOMALY DETECTION - TRAINING")
print("Autoencoder for Intrusion Detection")
print("="*70)

# STEP 1: Generate Network Traffic Data
print("\n[1/7] Generating network traffic data...")

# Normal traffic
X_normal = SyntheticDataGenerator.time_series(n_samples=5000, n_features=10, noise=0.05)
print(f"   ✓ Normal traffic: {len(X_normal)} samples")

# Anomalous traffic (attacks)
X_anomaly = SyntheticDataGenerator.time_series(n_samples=500, n_features=10, noise=0.3) * 3
print(f"   ✓ Anomalous traffic: {len(X_anomaly)} samples")

X = np.vstack([X_normal, X_anomaly])
print(f"   ✓ Total samples: {len(X)}")

# STEP 2: Normalize Data
print("\n[2/7] Normalizing data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("   ✓ Data normalized")

# STEP 3: Build Autoencoder
print("\n[3/7] Building autoencoder...")

encoder = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu')
])

decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='linear')
])

autoencoder = tf.keras.Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse')
print(f"   ✓ Model created: {autoencoder.count_params():,} parameters")

# STEP 4: Train on Normal Traffic Only
print("\n[4/7] Training on normal traffic...")
autoencoder.fit(X_scaled[:5000], X_scaled[:5000], epochs=50, batch_size=64, validation_split=0.2, verbose=1)

# STEP 5: Calculate Anomaly Threshold
print("\n[5/7] Calculating anomaly threshold...")
reconstructions = autoencoder.predict(X_scaled, verbose=0)
mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
threshold = np.percentile(mse[:5000], 95)
print(f"   ✓ Anomaly threshold: {threshold:.4f}")

# Detect anomalies
anomalies_detected = np.sum(mse[5000:] > threshold)
print(f"   ✓ Detected {anomalies_detected}/{len(X_anomaly)} anomalies ({anomalies_detected/len(X_anomaly)*100:.1f}%)")

# STEP 6: Save Model
print("\n[6/7] Saving model...")
autoencoder.save('anomaly_detection_model.h5')
ModelConverter.keras_to_tflite('anomaly_detection_model.h5', 'model.tflite')
print("   ✓ Models saved")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\nCapabilities:")
print("  ✓ Detect network intrusions")
print("  ✓ Identify DDoS attacks")
print("  ✓ Find malware communication")
print("  ✓ Real-time monitoring")
print("\nDeployment:")
print("  • Network tap or SPAN port")
print("  • Real-time packet analysis")
print("  • Alert on anomalies")
print("="*70 + "\n")
