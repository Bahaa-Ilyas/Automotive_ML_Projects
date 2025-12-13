# Architecture: Automotive Voice Assistant

## Pipeline
```
Speech → Whisper (ASR) → GPT-4 (NLU) → Vehicle API → TTS → Audio
```

## Components
1. **Whisper**: Speech recognition (1.5B params)
2. **Intent Classifier**: Categorize commands
3. **GPT-4**: Natural language understanding
4. **Vehicle API**: System integration
5. **TTS**: Text-to-speech synthesis

## Safety Features
- Voice-only interaction
- Emergency priority
- Manual override
- ISO 26262 ASIL-B

## Intents
- NAVIGATION, VEHICLE_STATUS, CLIMATE_CONTROL
- EMERGENCY, ENTERTAINMENT, DIAGNOSTICS
- DRIVING_ASSIST

## Next Steps
→ Project 34: Basic RAG with TF-IDF (Start RAG series)
