# Network Anomaly Detection Architecture

## System Overview
```
Network Traffic → Feature Extraction → Autoencoder → Anomaly Score → Alert
```

## Autoencoder Architecture
```
Input(10) → Dense(64) → Dense(32) → Dense(16) [Encoder]
Dense(16) → Dense(32) → Dense(64) → Output(10) [Decoder]
```

## Network Features
1. Packet size
2. Protocol type
3. Connection duration
4. Bytes sent/received
5. Port numbers
6. Packet rate
7. Error rate
8. Retransmission rate
9. Connection state
10. Service type

## Anomaly Detection
- Train on normal traffic only
- Reconstruction error > threshold → Anomaly
- Threshold: 95th percentile of normal traffic

## Deployment
- Network tap or SPAN port
- Real-time packet capture
- Feature extraction pipeline
- Model inference
- Alert system
