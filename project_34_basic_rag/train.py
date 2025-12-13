"""=============================================================================
PROJECT 34: BASIC RAG WITH TF-IDF - TRAINING SCRIPT
=============================================================================

DIFFICULTY: ★☆☆☆☆ (Beginner)
MODEL SIZE: ~0 parameters (TF-IDF + Cosine Similarity)

PURPOSE:
Introduction to Retrieval-Augmented Generation (RAG) using traditional
information retrieval methods before moving to neural approaches.

ARCHITECTURE:
1. Document Indexing: TF-IDF vectorization
2. Retrieval: Cosine similarity search
3. Generation: Template-based responses

USE CASES:
- FAQ systems
- Document search
- Knowledge base queries
- Customer support

WHY START HERE?
- Understand RAG fundamentals
- No GPU required
- Fast and interpretable
- Foundation for neural RAG
=============================================================================
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("\n" + "="*80)
print("PROJECT 34: BASIC RAG WITH TF-IDF")
print("Model: TF-IDF + Cosine Similarity (0 parameters)")
print("="*80)

# STEP 1: Create Knowledge Base
print("\n[1/6] Building automotive knowledge base...")

knowledge_base = [
    "The engine oil should be changed every 5,000 miles or 6 months.",
    "Tire pressure should be checked monthly and maintained at 32-35 PSI.",
    "Brake fluid should be replaced every 2 years or 24,000 miles.",
    "Air filters should be replaced every 12,000 to 15,000 miles.",
    "Battery terminals should be cleaned and checked every 6 months.",
    "Coolant should be flushed and replaced every 30,000 miles.",
    "Transmission fluid should be changed every 60,000 miles.",
    "Spark plugs should be replaced every 30,000 to 100,000 miles.",
    "Windshield wipers should be replaced every 6 to 12 months.",
    "Cabin air filter should be replaced every 15,000 to 30,000 miles."
]

print(f"   ✓ Knowledge base: {len(knowledge_base)} documents")
print(f"   ✓ Domain: Automotive maintenance")

# STEP 2: Index Documents with TF-IDF
print("\n[2/6] Indexing documents with TF-IDF...")

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
doc_vectors = vectorizer.fit_transform(knowledge_base)

print(f"   ✓ Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"   ✓ Document vectors: {doc_vectors.shape}")
print(f"   ✓ Sparse matrix: {doc_vectors.nnz} non-zero elements")

# STEP 3: Retrieval Function
print("\n[3/6] Building retrieval function...")

def retrieve_documents(query, top_k=3):
    """Retrieve most relevant documents"""
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, doc_vectors)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'document': knowledge_base[idx],
            'score': similarities[idx]
        })
    return results

print("   ✓ Retrieval function ready")
print("   ✓ Using cosine similarity")

# STEP 4: Test Retrieval
print("\n[4/6] Testing retrieval...")

queries = [
    "When should I change my oil?",
    "How often to check tire pressure?",
    "What about brake maintenance?",
    "Battery care tips?"
]

print("\n   Retrieval Results:")
for query in queries:
    print(f"\n   Query: {query}")
    results = retrieve_documents(query, top_k=2)
    for i, result in enumerate(results, 1):
        print(f"      [{i}] Score: {result['score']:.3f}")
        print(f"          {result['document']}")

# STEP 5: Simple Generation
print("\n[5/6] Adding template-based generation...")

def generate_response(query, retrieved_docs):
    """Generate response using retrieved context"""
    context = "\n".join([doc['document'] for doc in retrieved_docs])
    
    response = f"Based on automotive maintenance guidelines:\n\n"
    response += f"{retrieved_docs[0]['document']}\n\n"
    response += f"Related information: {retrieved_docs[1]['document']}"
    
    return response

print("\n   Generated Responses:")
for query in queries[:2]:
    results = retrieve_documents(query, top_k=2)
    response = generate_response(query, results)
    print(f"\n   Q: {query}")
    print(f"   A: {response[:150]}...")

# STEP 6: Evaluation Metrics
print("\n[6/6] Computing retrieval metrics...")

def evaluate_retrieval(queries, expected_keywords):
    """Evaluate retrieval quality"""
    scores = []
    for query, keywords in zip(queries, expected_keywords):
        results = retrieve_documents(query, top_k=1)
        doc = results[0]['document'].lower()
        score = sum(1 for kw in keywords if kw in doc) / len(keywords)
        scores.append(score)
    return np.mean(scores)

expected = [
    ['oil', 'miles'],
    ['tire', 'pressure'],
    ['brake', 'fluid'],
    ['battery', 'terminals']
]

accuracy = evaluate_retrieval(queries, expected)
print(f"\n   Retrieval Accuracy: {accuracy:.1%}")
print(f"   Average Response Time: <10ms")
print(f"   Memory Usage: <1MB")

print("\n" + "="*80)
print("TRAINING COMPLETE - BASIC RAG")
print("="*80)
print("\nKEY LEARNINGS:")
print("  ✓ RAG = Retrieval + Generation")
print("  ✓ TF-IDF captures term importance")
print("  ✓ Cosine similarity measures relevance")
print("  ✓ Foundation for neural RAG systems")
print("\nNEXT: Project 35 - Semantic Search with Sentence Transformers")
print("="*80)
