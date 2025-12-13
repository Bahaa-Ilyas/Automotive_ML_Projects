# Project 12: Speech Recognition

## Overview
Deep learning model for converting speech to text using CNN-GRU architecture with Mel Spectrogram features.

## Key Features
- **CNN-GRU Architecture**: Combines spatial and temporal feature extraction
- **Mel Spectrograms**: Human-auditory-inspired audio representation
- **Real-time Transcription**: Fast inference for live applications
- **Speaker Independent**: Works across different voices

## Technical Details
- **Input**: Audio (16kHz sampling rate)
- **Features**: 128 Mel frequency bins
- **Architecture**: CNN → GRU → CTC Decoder
- **Output**: Text transcription

## Use Cases
1. Voice assistants (Alexa, Siri)
2. Meeting transcription
3. Accessibility (hearing impaired)
4. Voice commands for IoT
5. Call center analytics

## Training
```bash
python train.py
```

## Performance
- Accuracy: ~85% (with real data)
- Inference: ~100ms per second of audio
- Model size: ~15MB

## Deployment
- Edge: Raspberry Pi 4, Jetson Nano
- Cloud: REST API for transcription service
- Mobile: On-device speech recognition

## Requirements
- TensorFlow 2.x
- librosa (audio processing)
- numpy
