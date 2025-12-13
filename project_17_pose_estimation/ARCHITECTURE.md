# Pose Estimation Architecture

## Model Architecture
```
Image(256×256) → CNN → GlobalAvgPool → Dense(512) → Output(17×2)
```

## Keypoints (17 joints)
1. Nose
2-3. Eyes (left, right)
4-5. Ears (left, right)
6-7. Shoulders (left, right)
8-9. Elbows (left, right)
10-11. Wrists (left, right)
12-13. Hips (left, right)
14-15. Knees (left, right)
16-17. Ankles (left, right)

## Training
- Loss: MSE (Mean Squared Error)
- Metric: MAE (Mean Absolute Error in pixels)
- Batch size: 16
- Epochs: 30-50

## Deployment
- Mobile: TFLite for on-device inference
- Edge: Jetson Nano for real-time video
- Cloud: Batch processing for video analytics
