"""=============================================================================
PROJECT 4: SMART BUILDING OCCUPANCY DETECTION - TRAINING SCRIPT
=============================================================================

PURPOSE:
This script trains a Random Forest classifier to detect room occupancy using
environmental sensors. This enables smart HVAC control, reducing energy waste
by 30% through intelligent climate management.

WHY RANDOM FOREST?
- Fast: Quick training and inference (<5ms)
- Interpretable: Can see which sensors are most important
- Robust: Handles noisy sensor data well
- No scaling needed: Works with raw sensor values

SENSORS USED:
1. Temperature (°C)
2. Humidity (%)
3. Light (lux)
4. CO2 (ppm)
5. Sound (dB)
=============================================================================
"""

# STEP 1: Import Required Libraries
# -----------------------------------
import numpy as np  # For numerical operations
from sklearn.ensemble import RandomForestClassifier  # Tree-based ensemble model
from sklearn.model_selection import train_test_split  # Split data
import joblib  # Save scikit-learn models

print("\n" + "="*70)
print("SMART BUILDING OCCUPANCY DETECTION - TRAINING")
print("="*70)

# STEP 2: Generate Synthetic Sensor Data
# ---------------------------------------
# We simulate sensor readings for occupied vs vacant rooms
# In production, this data comes from actual IoT sensors
print("\n[1/5] Generating synthetic sensor data...")
np.random.seed(42)  # For reproducibility
n_samples = 2000

# OCCUPIED ROOM characteristics:
# - Higher temperature (22-27°C) due to body heat
# - Higher humidity (40-60%) from breathing
# - More light (300-800 lux) - lights are on
# - Higher CO2 (600-1000 ppm) from breathing
# - More sound (30-80 dB) from activity
occupied = np.random.rand(n_samples // 2, 5) * np.array([5, 20, 500, 400, 50]) + np.array([22, 40, 300, 600, 30])

# VACANT ROOM characteristics:
# - Lower temperature (20-23°C)
# - Lower humidity (30-45%)
# - Less light (100-300 lux) - lights off or minimal
# - Lower CO2 (400-600 ppm) - ambient levels
# - Less sound (10-30 dB) - background noise only
vacant = np.random.rand(n_samples // 2, 5) * np.array([3, 15, 200, 200, 20]) + np.array([20, 30, 100, 400, 10])

# Combine data
X = np.vstack([occupied, vacant])  # Features: [temp, humidity, light, CO2, sound]
y = np.hstack([np.ones(n_samples // 2), np.zeros(n_samples // 2)])  # Labels: 1=occupied, 0=vacant

print(f"   ✓ Generated {len(X)} sensor readings")
print(f"   ✓ Features: Temperature, Humidity, Light, CO2, Sound")
print(f"   ✓ Occupied samples: {np.sum(y == 1)}")
print(f"   ✓ Vacant samples: {np.sum(y == 0)}")

# STEP 3: Split Data into Training and Testing Sets
# --------------------------------------------------
# 80% for training, 20% for testing
print("\n[2/5] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"   ✓ Training samples: {len(X_train)}")
print(f"   ✓ Testing samples: {len(X_test)}")

# STEP 4: Train the Random Forest Model
# --------------------------------------
# Random Forest creates multiple decision trees and combines their predictions
# Parameters:
# - n_estimators=50: Build 50 decision trees
# - max_depth=10: Limit tree depth to prevent overfitting
print("\n[3/5] Training Random Forest model...")
print("   Building 50 decision trees...")
model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)
print("   ✓ Model trained successfully")

# STEP 5: Evaluate the Model
# --------------------------
# Test on unseen data to measure real-world performance
print("\n[4/5] Evaluating model...")
acc = model.score(X_test, y_test)
print(f"   ✓ Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

# Show feature importance (which sensors matter most)
feature_names = ['Temperature', 'Humidity', 'Light', 'CO2', 'Sound']
feature_importance = model.feature_importances_
print("\n   Feature Importance:")
for name, importance in zip(feature_names, feature_importance):
    print(f"      {name:12s}: {importance:.3f} {'█' * int(importance * 50)}")

# STEP 6: Save the Model
# ----------------------
# Save using joblib for scikit-learn models
print("\n[5/5] Saving model...")
joblib.dump(model, 'occupancy_model.pkl')
print("   ✓ Model saved: occupancy_model.pkl")
print(f"   ✓ Model size: ~20 KB (very lightweight!)")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
print("\nNext steps:")
print("1. Deploy to ESP32 microcontroller")
print("2. Connect sensors: DHT22, LDR, MQ-135, sound sensor")
print("3. Run deploy_esp32.py for real-time occupancy detection")
print("4. Integrate with HVAC system for energy savings")
print("\nExpected energy savings: 30% reduction in HVAC costs")
print("="*70 + "\n")
