"""=============================================================================
PROJECT 24: NAMED ENTITY RECOGNITION (NER) - TRAINING SCRIPT
=============================================================================

DIFFICULTY: ★★☆☆☆ (Beginner-Intermediate)
NLP MODEL: spaCy (Small, Pre-trained)

PURPOSE:
Extract named entities (people, organizations, locations) from text.
Uses pre-trained spaCy model for entity recognition.

TECHNIQUE: spaCy NER
- Pre-trained on large corpus
- Recognizes: PERSON, ORG, GPE, DATE, MONEY, etc.
- Fast inference
- Production-ready

USE CASES:
- Information extraction
- Document analysis
- Customer data extraction
- News article processing

WHY spaCy?
- Pre-trained models available
- Fast and accurate
- Easy to use
- Industry standard
=============================================================================
"""

import spacy
from spacy.training import Example
import random

print("\n" + "="*80)
print("PROJECT 24: NAMED ENTITY RECOGNITION")
print("spaCy NER Model")
print("="*80)

# STEP 1: Load Pre-trained Model
print("\n[1/6] Loading spaCy model...")
try:
    nlp = spacy.load("en_core_web_sm")
    print("   ✓ Loaded: en_core_web_sm")
except:
    print("   ⚠ Model not found. Installing...")
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    print("   ✓ Installed and loaded: en_core_web_sm")

# STEP 2: Test Pre-trained Model
print("\n[2/6] Testing pre-trained NER...")
test_texts = [
    "Apple Inc. is headquartered in Cupertino, California.",
    "Elon Musk founded Tesla and SpaceX in the United States.",
    "The meeting is scheduled for January 15, 2024 in New York."
]

print("\n   Pre-trained Model Results:")
for text in test_texts:
    doc = nlp(text)
    print(f"\n   Text: {text}")
    for ent in doc.ents:
        print(f"      {ent.text:20s} → {ent.label_:10s}")

# STEP 3: Create Custom Training Data
print("\n[3/6] Creating custom training data...")

TRAIN_DATA = [
    ("BMW is a German automotive company.", {"entities": [(0, 3, "ORG"), (9, 15, "NORP")]}),
    ("Mercedes-Benz manufactures luxury vehicles.", {"entities": [(0, 13, "ORG")]}),
    ("Tesla was founded by Elon Musk in 2003.", {"entities": [(0, 5, "ORG"), (21, 30, "PERSON"), (34, 38, "DATE")]}),
    ("Volkswagen Group is based in Wolfsburg, Germany.", {"entities": [(0, 16, "ORG"), (29, 38, "GPE"), (40, 47, "GPE")]}),
    ("Toyota produces cars in Japan.", {"entities": [(0, 6, "ORG"), (24, 29, "GPE")]}),
]

print(f"   ✓ Created {len(TRAIN_DATA)} training examples")
print("   ✓ Entities: ORG, PERSON, GPE, DATE, NORP")

# STEP 4: Fine-tune Model (Optional)
print("\n[4/6] Fine-tuning on custom data...")

# Get the NER component
ner = nlp.get_pipe("ner")

# Add new labels if needed
for _, annotations in TRAIN_DATA:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Disable other pipes during training
other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
with nlp.disable_pipes(*other_pipes):
    optimizer = nlp.resume_training()
    
    # Training loop
    for iteration in range(10):
        random.shuffle(TRAIN_DATA)
        losses = {}
        
        for text, annotations in TRAIN_DATA:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], drop=0.5, losses=losses)
        
        if iteration % 5 == 0:
            print(f"   Iteration {iteration}: Loss = {losses['ner']:.2f}")

print("   ✓ Fine-tuning complete")

# STEP 5: Evaluate Fine-tuned Model
print("\n[5/6] Evaluating fine-tuned model...")

test_automotive = [
    "Ford Motor Company is based in Detroit, Michigan.",
    "Audi is a subsidiary of Volkswagen Group.",
    "Honda manufactures motorcycles and automobiles in Tokyo."
]

print("\n   Fine-tuned Model Results:")
for text in test_automotive:
    doc = nlp(text)
    print(f"\n   Text: {text}")
    for ent in doc.ents:
        print(f"      {ent.text:20s} → {ent.label_:10s}")

# STEP 6: Save Model
print("\n[6/6] Saving fine-tuned model...")
nlp.to_disk("./ner_model")
print("   ✓ Model saved: ./ner_model")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nKey Learnings:")
print("  ✓ Pre-trained NER models")
print("  ✓ Entity types: PERSON, ORG, GPE, DATE")
print("  ✓ Fine-tuning on custom data")
print("  ✓ Fast inference (<1ms per sentence)")
print("\nEntity Types Recognized:")
print("  • PERSON: People names")
print("  • ORG: Organizations, companies")
print("  • GPE: Countries, cities, states")
print("  • DATE: Dates and times")
print("  • MONEY: Monetary values")
print("\nNext Level: Project 25 - Text Summarization")
print("="*80 + "\n")
