"""=============================================================================
PROJECT 19: IMAGE SEGMENTATION - TRAINING SCRIPT
=============================================================================

PURPOSE:
Train a U-Net model for semantic image segmentation - classifying every pixel
in an image. Used in medical imaging, autonomous vehicles, satellite imagery,
and industrial inspection.

WHY U-NET?
- Pixel-level Classification: Precise object boundaries
- Skip Connections: Preserve spatial information
- Efficient: Works with small datasets
- State-of-the-art: Best for medical and industrial imaging

SEGMENTATION CLASSES:
- Background
- Object 1 (e.g., road, tumor, defect)
- Object 2 (e.g., vehicle, organ, product)

USE CASES:
- Medical imaging (tumor segmentation)
- Autonomous vehicles (road/obstacle segmentation)
- Satellite imagery (land use classification)
- Industrial inspection (defect segmentation)
- Agriculture (crop/weed segmentation)
=============================================================================
"""

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import sys
sys.path.append('..')
from shared_utils.data_generator import SyntheticDataGenerator
from shared_utils.model_converter import ModelConverter

print("\n" + "="*70)
print("IMAGE SEGMENTATION - TRAINING")
print("U-Net for Pixel-level Classification")
print("="*70)

# STEP 1: Generate Synthetic Segmentation Data
print("\n[1/6] Generating synthetic segmentation data...")

# Images
X, _ = SyntheticDataGenerator.image_data(n_samples=500, img_size=(256, 256, 3), n_classes=1)

# Segmentation masks (each pixel has a class label)
y = np.random.randint(0, 3, (500, 256, 256, 1)).astype(np.float32)  # 3 classes

print(f"   ✓ Generated {len(X)} images")
print(f"   ✓ Image size: 256×256")
print(f"   ✓ Segmentation classes: 3")
print(f"   ✓ Output: Pixel-level classification")

# STEP 2: Split Data
print("\n[2/6] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   ✓ Training: {len(X_train)}")
print(f"   ✓ Testing: {len(X_test)}")

# STEP 3: Build U-Net Model
print("\n[3/6] Building U-Net model...")

def unet_model():
    inputs = tf.keras.layers.Input(shape=(256, 256, 3))
    
    # Encoder (downsampling)
    c1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    p1 = tf.keras.layers.MaxPooling2D(2)(c1)
    
    c2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(p1)
    p2 = tf.keras.layers.MaxPooling2D(2)(c2)
    
    # Bottleneck
    c3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(p2)
    
    # Decoder (upsampling)
    u1 = tf.keras.layers.UpSampling2D(2)(c3)
    u1 = tf.keras.layers.Concatenate()([u1, c2])  # Skip connection
    c4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(u1)
    
    u2 = tf.keras.layers.UpSampling2D(2)(c4)
    u2 = tf.keras.layers.Concatenate()([u2, c1])  # Skip connection
    c5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
    
    # Output
    outputs = tf.keras.layers.Conv2D(3, 1, activation='softmax')(c5)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

model = unet_model()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(f"   ✓ Model created: {model.count_params():,} parameters")
print("   ✓ Architecture: U-Net with skip connections")

# STEP 4: Train
print("\n[4/6] Training model...")
history = model.fit(
    X_train, y_train,
    epochs=25,
    batch_size=8,
    validation_split=0.2,
    verbose=1
)

# STEP 5: Evaluate
print("\n[5/6] Evaluating model...")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"   ✓ Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"   ✓ Pixel-level accuracy")

# STEP 6: Save Model
print("\n[6/6] Saving model...")
model.save('image_segmentation_model.h5')
ModelConverter.keras_to_tflite('image_segmentation_model.h5', 'model.tflite')
print("   ✓ Models saved")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\nCapabilities:")
print("  ✓ Pixel-level image segmentation")
print("  ✓ Precise object boundaries")
print("  ✓ Multi-class segmentation")
print("\nApplications:")
print("  • Medical imaging (tumor segmentation)")
print("  • Autonomous vehicles (road segmentation)")
print("  • Satellite imagery (land use)")
print("  • Industrial inspection (defect detection)")
print("\nDeployment:")
print("  • Edge: Jetson Nano for real-time")
print("  • Cloud: Batch processing for medical images")
print("="*70 + "\n")
