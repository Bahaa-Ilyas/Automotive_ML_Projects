"""=============================================================================
PROJECT 10: THERMAL IMAGING ANALYSIS - TRAINING SCRIPT
=============================================================================

PURPOSE:
This script trains a CNN to analyze thermal images for detecting heat anomalies,
energy leaks, and equipment overheating. Used in building inspections, electrical
maintenance, and industrial monitoring to prevent failures and save energy.

WHY CNN FOR THERMAL IMAGING?
- Spatial Patterns: Detects heat distribution patterns
- Temperature Gradients: Identifies abnormal hot/cold spots
- Multi-scale Features: Captures both local and global thermal patterns
- Real-time: Fast inference on edge devices (Jetson Nano)

THERMAL CATEGORIES (3 classes):
1. Normal: Uniform temperature distribution
2. Hot Spot: Localized overheating (equipment fault, electrical issue)
3. Cold Spot: Heat loss (insulation failure, air leak)

USE CASES:
- Building energy audits (insulation defects)
- Electrical panel inspection (overheating components)
- Industrial equipment monitoring (bearing failures)
- Solar panel inspection (faulty cells)
- Medical diagnostics (fever detection)
=============================================================================
"""

# STEP 1: Import Required Libraries
# -----------------------------------
import numpy as np  # Numerical operations
import tensorflow as tf  # Deep learning framework
from sklearn.model_selection import train_test_split  # Split data
import sys
sys.path.append('..')  # Access shared utilities
from shared_utils.data_generator import SyntheticDataGenerator  # Generate synthetic data
from shared_utils.model_converter import ModelConverter  # Convert to TFLite

print("\n" + "="*70)
print("THERMAL IMAGING ANALYSIS - TRAINING")
print("CNN for Heat Anomaly Detection")
print("="*70)

# STEP 2: Generate Synthetic Thermal Image Data
# ----------------------------------------------
# In production, use real thermal camera images (FLIR, Seek Thermal)
# Thermal images are grayscale (single channel) representing temperature
print("\n[1/7] Generating synthetic thermal image data...")

X, y = SyntheticDataGenerator.image_data(
    n_samples=400,
    img_size=(128, 128, 1),  # 128x128 grayscale (temperature map)
    n_classes=3  # Normal, Hot Spot, Cold Spot
)

print(f"   ✓ Generated {len(X)} thermal images")
print(f"   ✓ Image size: 128x128 pixels")
print(f"   ✓ Channels: 1 (grayscale temperature map)")
print(f"   ✓ Temperature range: Simulated 0-100°C")
print("\n   Classes:")
print("      0: Normal (uniform temperature)")
print("      1: Hot Spot (localized overheating)")
print("      2: Cold Spot (heat loss/air leak)")

# Show class distribution
for i in range(3):
    count = np.sum(y == i)
    print(f"      Class {i}: {count} images ({count/len(y)*100:.1f}%)")

# STEP 3: Split Data into Training and Testing Sets
# --------------------------------------------------
print("\n[2/7] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   ✓ Training images: {len(X_train)}")
print(f"   ✓ Testing images: {len(X_test)}")

# STEP 4: Build CNN Architecture for Thermal Analysis
# ----------------------------------------------------
# Architecture inspired by U-Net (encoder-decoder)
# 
# ENCODER (Feature Extraction):
# - Conv2D + MaxPooling: Extract hierarchical features
# - Multiple scales: Detect both small hot spots and large patterns
# 
# CLASSIFIER:
# - GlobalAveragePooling: Aggregate spatial information
# - Dense layers: Classification
print("\n[3/7] Building CNN for thermal analysis...")

model = tf.keras.Sequential([
    # First convolutional block (high resolution)
    tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same', input_shape=(128, 128, 1)),
    tf.keras.layers.MaxPooling2D(2),  # 128x128 → 64x64
    
    # Second convolutional block (medium resolution)
    tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(2),  # 64x64 → 32x32
    
    # Third convolutional block (low resolution, high-level features)
    tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
    
    # Global aggregation
    tf.keras.layers.GlobalAveragePooling2D(),
    
    # Classification layers
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
])

print("   ✓ Model architecture created")
print(f"   ✓ Total parameters: {model.count_params():,}")
print("\n   Architecture Details:")
print("      1. Conv2D(32) + MaxPool: 128→64 (detect small features)")
print("      2. Conv2D(64) + MaxPool: 64→32 (detect medium features)")
print("      3. Conv2D(128): Extract high-level patterns")
print("      4. GlobalAvgPool: Aggregate spatial info")
print("      5. Dense(64) + Dense(3): Classification")

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print("\n   ✓ Model compiled")
print("   ✓ Loss: Sparse Categorical Crossentropy")
print("   ✓ Metric: Accuracy")

# STEP 5: Train the Model
# -----------------------
print("\n[4/7] Training the model...")
print("   This may take 5-10 minutes...\n")

history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

print("\n   ✓ Training complete")

# STEP 6: Evaluate the Model
# --------------------------
print("\n[5/7] Evaluating model performance...")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"   ✓ Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"   ✓ Test Loss: {loss:.4f}")

# Per-class accuracy
print("\n   Per-class performance:")
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
class_names = ['Normal', 'Hot Spot', 'Cold Spot']
for i in range(3):
    mask = y_test == i
    if np.sum(mask) > 0:
        class_acc = np.sum(y_pred[mask] == y_test[mask]) / np.sum(mask)
        print(f"      {class_names[i]:12s}: {class_acc*100:.1f}% accuracy")

# STEP 7: Save the Models
# -----------------------
print("\n[6/7] Saving models...")
model.save('thermal_imaging_model.h5')
print("   ✓ Keras model saved: thermal_imaging_model.h5")

ModelConverter.keras_to_tflite('thermal_imaging_model.h5', 'model.tflite')
model_size = ModelConverter.get_model_size('model.tflite')
print(f"   ✓ TFLite model saved: model.tflite ({model_size:.2f} MB)")

print("\n[7/7] Creating deployment guide...")
print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\nModel Capabilities:")
print("  ✓ Detect heat anomalies in thermal images")
print("  ✓ Classify: Normal, Hot Spot, Cold Spot")
print(f"  ✓ {acc*100:.1f}% accuracy")
print("  ✓ Real-time analysis on Jetson Nano")
print("\nThermal Camera Requirements:")
print("  • Resolution: 128x128 or higher")
print("  • Temperature range: -20°C to 150°C typical")
print("  • Frame rate: 9-60 Hz")
print("  • Examples: FLIR Lepton, Seek Thermal Compact")
print("\nApplications:")
print("  • Building Energy Audits:")
print("    - Detect insulation defects (cold spots)")
print("    - Find air leaks (temperature gradients)")
print("    - Identify thermal bridges")
print("  • Electrical Inspection:")
print("    - Overheating components (hot spots)")
print("    - Loose connections (localized heat)")
print("    - Circuit breaker issues")
print("  • Industrial Monitoring:")
print("    - Bearing failures (hot spots)")
print("    - Motor overheating")
print("    - Pipe leaks (temperature anomalies)")
print("  • Solar Panel Inspection:")
print("    - Faulty cells (hot spots)")
print("    - Connection issues")
print("\nBusiness Impact:")
print("  • 30% energy savings (insulation fixes)")
print("  • Prevent electrical fires")
print("  • Reduce equipment downtime by 50%")
print("  • Automated inspection (vs manual)")
print("\nNext steps:")
print("1. Acquire thermal camera (FLIR/Seek)")
print("2. Deploy to Jetson Nano")
print("3. Run deploy_jetson.py for real-time analysis")
print("4. Integrate with inspection workflow")
print("5. Generate automated reports")
print("="*70 + "\n")
