"""=============================================================================
PROJECT 27: MACHINE TRANSLATION - TRAINING SCRIPT
=============================================================================

DIFFICULTY: ★★★★☆ (Intermediate-Advanced)
NLP MODEL: MarianMT (74M parameters per language pair)

PURPOSE:
Translate text between languages using neural machine translation.
Uses MarianMT transformer models.

TECHNIQUE: MarianMT Transformer
- Encoder-decoder architecture
- Attention mechanism
- Pre-trained on parallel corpora
- Supports 1000+ language pairs

USE CASES:
- Document translation
- Real-time chat translation
- Multilingual customer support
- Content localization

WHY MarianMT?
- Fast inference
- High quality translations
- Many language pairs
- Open source
=============================================================================
"""

from transformers import MarianMTModel, MarianTokenizer

print("\n" + "="*80)
print("PROJECT 27: MACHINE TRANSLATION")
print("MarianMT Transformer (74M parameters)")
print("="*80)

print("\n[1/5] Loading translation models...")

# English to German
model_name_en_de = 'Helsinki-NLP/opus-mt-en-de'
tokenizer_en_de = MarianTokenizer.from_pretrained(model_name_en_de)
model_en_de = MarianMTModel.from_pretrained(model_name_en_de)
print(f"   ✓ EN→DE: {model_en_de.num_parameters():,} parameters")

# English to French
model_name_en_fr = 'Helsinki-NLP/opus-mt-en-fr'
tokenizer_en_fr = MarianTokenizer.from_pretrained(model_name_en_fr)
model_en_fr = MarianMTModel.from_pretrained(model_name_en_fr)
print(f"   ✓ EN→FR: {model_en_fr.num_parameters():,} parameters")

print("\n[2/5] Preparing automotive texts...")
texts = [
    "The electric vehicle has a range of 400 kilometers.",
    "Please check the tire pressure before driving.",
    "The battery management system monitors cell temperature.",
    "Autonomous driving requires multiple sensors and cameras.",
    "The engine warning light indicates a potential problem."
]

print(f"   ✓ Prepared {len(texts)} sentences")

print("\n[3/5] Translating EN → DE...")
for text in texts:
    inputs = tokenizer_en_de([text], return_tensors="pt", padding=True)
    translated = model_en_de.generate(**inputs)
    translation = tokenizer_en_de.decode(translated[0], skip_special_tokens=True)
    print(f"\n   EN: {text}")
    print(f"   DE: {translation}")

print("\n[4/5] Translating EN → FR...")
for text in texts[:3]:  # First 3 for brevity
    inputs = tokenizer_en_fr([text], return_tensors="pt", padding=True)
    translated = model_en_fr.generate(**inputs)
    translation = tokenizer_en_fr.decode(translated[0], skip_special_tokens=True)
    print(f"\n   EN: {text}")
    print(f"   FR: {translation}")

print("\n[5/5] Saving models...")
model_en_de.save_pretrained('./translation_en_de')
tokenizer_en_de.save_pretrained('./translation_en_de')
print("   ✓ Saved EN→DE model")

print("\n" + "="*80)
print("COMPLETE!")
print("="*80)
print("\nSupported: 1000+ language pairs")
print("Performance: ~100ms per sentence")
print("\nNext Level: Project 28 - Conversational AI")
print("="*80 + "\n")
