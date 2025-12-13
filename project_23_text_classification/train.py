"""=============================================================================
PROJECT 23: BASIC TEXT CLASSIFICATION - TRAINING SCRIPT
=============================================================================

DIFFICULTY: ★☆☆☆☆ (Beginner)
NLP MODEL: Bag-of-Words + Naive Bayes (Simple, Fast)

PURPOSE:
Learn fundamentals of text classification using traditional NLP methods.
Classify text into categories without deep learning.

TECHNIQUE: TF-IDF + Naive Bayes
- TF-IDF: Term Frequency-Inverse Document Frequency
- Naive Bayes: Probabilistic classifier
- No neural networks needed
- Fast training and inference

USE CASES:
- Spam detection
- Topic classification
- Sentiment analysis (basic)
- Document categorization

WHY START HERE?
- Understand text preprocessing
- Learn feature extraction
- Baseline for comparison
- Production-ready for simple tasks
=============================================================================
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

print("\n" + "="*80)
print("PROJECT 23: BASIC TEXT CLASSIFICATION")
print("TF-IDF + Naive Bayes")
print("="*80)

# STEP 1: Generate Sample Text Data
print("\n[1/6] Generating sample text data...")

# Sample texts for 3 categories
texts = [
    # Technology
    "artificial intelligence machine learning deep neural networks",
    "computer programming software development coding python",
    "data science analytics big data processing",
    "cloud computing servers infrastructure deployment",
    "cybersecurity encryption network protection firewall",
    # Sports
    "football soccer match goal championship tournament",
    "basketball game score points team victory",
    "tennis player serve match championship",
    "olympics athletes competition medals gold silver",
    "training fitness exercise workout performance",
    # Business
    "stock market investment trading portfolio profit",
    "company revenue growth sales marketing strategy",
    "management leadership team organization corporate",
    "finance banking loans credit interest rates",
    "startup entrepreneur business plan funding venture"
] * 20  # Repeat for more samples

labels = [0]*100 + [1]*100 + [2]*100  # 0=Tech, 1=Sports, 2=Business

print(f"   ✓ Generated {len(texts)} text samples")
print(f"   ✓ Categories: 3 (Technology, Sports, Business)")
print(f"   ✓ Samples per category: {len(texts)//3}")

# STEP 2: Split Data
print("\n[2/6] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)
print(f"   ✓ Training: {len(X_train)} samples")
print(f"   ✓ Testing: {len(X_test)} samples")

# STEP 3: Feature Extraction with TF-IDF
print("\n[3/6] Extracting TF-IDF features...")
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"   ✓ Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"   ✓ Feature matrix shape: {X_train_tfidf.shape}")
print("   ✓ TF-IDF: Weighs words by importance")

# STEP 4: Train Naive Bayes Classifier
print("\n[4/6] Training Naive Bayes classifier...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
print("   ✓ Model trained")
print("   ✓ Training time: <1 second")

# STEP 5: Evaluate
print("\n[5/6] Evaluating model...")
y_pred = model.predict(X_test_tfidf)
acc = model.score(X_test_tfidf, y_test)

print(f"   ✓ Accuracy: {acc*100:.1f}%")
print("\n   Classification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['Technology', 'Sports', 'Business']))

# STEP 6: Save Model
print("\n[6/6] Saving model...")
joblib.dump(model, 'text_classifier.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
print("   ✓ Model saved: text_classifier.pkl")
print("   ✓ Vectorizer saved: tfidf_vectorizer.pkl")

# Test with new text
print("\n" + "="*80)
print("TESTING WITH NEW TEXT")
print("="*80)
test_texts = [
    "python programming neural networks",
    "football championship final match",
    "stock market investment strategy"
]

for text in test_texts:
    text_tfidf = vectorizer.transform([text])
    pred = model.predict(text_tfidf)[0]
    prob = model.predict_proba(text_tfidf)[0]
    categories = ['Technology', 'Sports', 'Business']
    print(f"\nText: '{text}'")
    print(f"Predicted: {categories[pred]} ({prob[pred]*100:.1f}% confidence)")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print("\nKey Learnings:")
print("  ✓ Text preprocessing and tokenization")
print("  ✓ TF-IDF feature extraction")
print("  ✓ Naive Bayes classification")
print("  ✓ Fast training (<1 second)")
print("  ✓ Good baseline for text classification")
print("\nNext Level: Project 24 - Named Entity Recognition")
print("="*80 + "\n")
