"""=============================================================================
PROJECT 2: QUALITY CONTROL VISION SYSTEM - TRAINING SCRIPT
=============================================================================

PURPOSE:
This script trains a computer vision model to automatically detect defects in
manufactured products. It uses MobileNetV2, a lightweight CNN architecture
optimized for edge devices like NVIDIA Jetson Nano.

WHY MobileNetV2?
- Efficient: Uses depthwise separable convolutions (fewer parameters)
- Fast: Optimized for mobile and edge devices
- Accurate: Achieves high accuracy despite being lightweight

USE CASE:
Automated quality inspection in manufacturing lines, replacing manual inspection
and reducing human error.
=============================================================================
"""

# STEP 1: Import Required Libraries
# -----------------------------------
import tensorflow as tf  # Deep learning framework
from tensorflow.keras.applications import MobileNetV2  # Pre-built efficient CNN architecture
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D  # Additional layers
import sys
sys.path.append('..')  # Access shared utilities
from shared_utils.data_generator import SyntheticDataGenerator  # Generate synthetic images
from shared_utils.model_converter import ModelConverter  # Convert to TFLite

print("\n" + "="*70)
print("QUALITY CONTROL VISION SYSTEM - TRAINING")
print("="*70)

# STEP 2: Generate Synthetic Image Data
# --------------------------------------
# In production, these would be real images of products
# We generate 500 RGB images (224x224 pixels) with 2 classes:
# - Class 0: OK (no defects)
# - Class 1: Defect detected
print("\n[1/5] Generating synthetic image data...")
X, y = SyntheticDataGenerator.image_data(n_samples=500, img_size=(224, 224, 3), n_classes=2)
print(f"   ✓ Generated {len(X)} images")
print(f"   ✓ Image shape: {X.shape[1:]}")
print(f"   ✓ Classes: 0=OK, 1=Defect")
print(f"   ✓ Class distribution: OK={np.sum(y==0)}, Defect={np.sum(y==1)}")

# STEP 3: Split Data into Training and Testing Sets
# --------------------------------------------------
# 80% for training, 20% for testing
# Simple split (no shuffle) for demonstration
print("\n[2/5] Splitting data...")
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print(f"   ✓ Training images: {len(X_train)}")
print(f"   ✓ Testing images: {len(X_test)}")

# STEP 4: Build the MobileNetV2 Model
# ------------------------------------
# Architecture:
# - Base: MobileNetV2 (efficient CNN backbone)
# - GlobalAveragePooling: Reduces spatial dimensions
# - Dense(64): Feature extraction layer
# - Dense(2): Output layer for 2 classes (OK/Defect)
print("\n[3/5] Building MobileNetV2 model...")

# Load MobileNetV2 without pre-trained weights and without top classification layer
base_model = MobileNetV2(weights=None, include_top=False, input_shape=(224, 224, 3))
print("   ✓ MobileNetV2 base loaded")

# Add custom classification head
x = base_model.output  # Get output from MobileNetV2
x = GlobalAveragePooling2D()(x)  # Convert feature maps to single vector
x = Dense(64, activation='relu')(x)  # Add dense layer for feature learning
predictions = Dense(2, activation='softmax')(x)  # Output layer: 2 classes with probabilities

# Create the complete model
model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
print(f"   ✓ Custom head added")
print(f"   ✓ Total parameters: {model.count_params():,}")

# Compile the model
# - Adam optimizer: adaptive learning rate
# - Sparse categorical crossentropy: for integer labels (0, 1)
# - Accuracy: metric to monitor
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("   ✓ Model compiled")

# STEP 5: Train the Model
# -----------------------
# Train for 10 epochs with small batch size (16) due to image size
# 20% of training data used for validation
print("\n[4/5] Training the model...")
print("   This may take 5-10 minutes...\n")
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# STEP 6: Evaluate the Model
# --------------------------
# Test on unseen images to measure real-world performance
print("\n[5/5] Evaluating model...")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"   ✓ Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"   ✓ Test Loss: {loss:.4f}")

# STEP 7: Save and Optimize the Model
# ------------------------------------
# Save in two formats:
# 1. Keras format (.h5) - full model
# 2. TFLite format with quantization - optimized for Jetson Nano
print("\n[6/6] Saving and optimizing model...")
model.save('quality_control_model.h5')
print("   ✓ Keras model saved: quality_control_model.h5")

# Convert to TFLite with INT8 quantization for faster inference
ModelConverter.keras_to_tflite('quality_control_model.h5', 'model.tflite', quantize=True)
model_size = ModelConverter.get_model_size('model.tflite')
print(f"   ✓ TFLite model saved: model.tflite ({model_size:.2f} MB)")
print("   ✓ Quantization applied (INT8) for faster inference")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\nNext steps:")
print("1. Deploy model.tflite to NVIDIA Jetson Nano")
print("2. Run deploy_jetson.py with camera feed")
print("3. Integrate with production line for real-time inspection")
print("="*70 + "\n")
