"""=============================================================================
PROJECT 18: TIME SERIES CLUSTERING - TRAINING SCRIPT
=============================================================================

PURPOSE:
Train an Autoencoder + K-Means model for clustering time series data. Discovers
patterns in sensor data, customer behavior, and operational metrics without
labeled data. Enables anomaly detection and pattern discovery.

WHY AUTOENCODER + K-MEANS?
- Dimensionality Reduction: Compress time series to compact representation
- Unsupervised: No labels needed
- Pattern Discovery: Find similar behaviors automatically
- Scalable: Handle thousands of time series

USE CASES:
- Customer segmentation (purchase patterns)
- Sensor pattern discovery (IoT devices)
- Load profiling (energy consumption patterns)
- Network traffic clustering
- Financial market regime detection
=============================================================================
"""

import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('..')
from shared_utils.data_generator import SyntheticDataGenerator

print("\n" + "="*70)
print("TIME SERIES CLUSTERING - TRAINING")
print("Autoencoder + K-Means for Pattern Discovery")
print("="*70)

# STEP 1: Generate Time Series Data
print("\n[1/7] Generating time series data...")

# Generate 3 different patterns
pattern1 = SyntheticDataGenerator.time_series(n_samples=500, n_features=100, noise=0.1)
pattern2 = SyntheticDataGenerator.time_series(n_samples=500, n_features=100, noise=0.1) * 2
pattern3 = SyntheticDataGenerator.time_series(n_samples=500, n_features=100, noise=0.1) * 0.5

X = np.vstack([pattern1, pattern2, pattern3])

print(f"   ✓ Generated {len(X)} time series")
print(f"   ✓ Length: 100 timesteps each")
print(f"   ✓ True patterns: 3 (for validation)")

# STEP 2: Normalize Data
print("\n[2/7] Normalizing data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("   ✓ Data normalized")

# STEP 3: Build Autoencoder
print("\n[3/7] Building autoencoder...")

encoder = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(100,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu')
])

decoder = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(16,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(100, activation='linear')
])

autoencoder = tf.keras.Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse')
print(f"   ✓ Model created: {autoencoder.count_params():,} parameters")

# STEP 4: Train Autoencoder
print("\n[4/7] Training autoencoder...")
history = autoencoder.fit(
    X_scaled, X_scaled,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# STEP 5: Extract Embeddings
print("\n[5/7] Extracting embeddings...")
embeddings = encoder.predict(X_scaled, verbose=0)
print(f"   ✓ Compressed: 100 → 16 dimensions")
print(f"   ✓ Compression ratio: {100/16:.1f}x")

# STEP 6: Cluster Embeddings
print("\n[6/7] Clustering time series...")

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings)

print(f"   ✓ Found {n_clusters} clusters")
for i in range(n_clusters):
    count = np.sum(clusters == i)
    print(f"      Cluster {i}: {count} time series ({count/len(X)*100:.1f}%)")

# STEP 7: Save Models
print("\n[7/7] Saving models...")
autoencoder.save('time_series_clustering_model.h5')
import joblib
joblib.dump(kmeans, 'kmeans_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("   ✓ Models saved")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\nCapabilities:")
print("  ✓ Discover patterns in time series")
print("  ✓ Unsupervised clustering")
print("  ✓ Dimensionality reduction (100 → 16)")
print("  ✓ Anomaly detection (outlier clusters)")
print("\nApplications:")
print("  • Customer segmentation")
print("  • Sensor pattern discovery")
print("  • Load profiling")
print("  • Network traffic analysis")
print("\nDeployment:")
print("  • Batch processing for large datasets")
print("  • Real-time classification of new time series")
print("="*70 + "\n")
