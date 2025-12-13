# Architecture: Text Summarization

## Model: BART
```
Input Text → Encoder (6 layers) → Decoder (6 layers) → Summary
```

## BART Features
- Seq2seq transformer
- Pre-trained on denoising tasks
- Abstractive summarization
- 140M parameters

## Summarization Types
- Extractive: Select key sentences
- Abstractive: Generate new text (BART)

## Next Steps
→ Project 26: Question Answering with DistilBERT (66M)
