import openai
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import dotenv_values

# Load environment variables from .env file
dotenv_values = dotenv_values(".env")

# Set your OpenAI API key (get it from platform.openai.com)
# openai.api_key = dotenv_values.get("api-key")

# 10 sentences: 5 pairs of similar + 1 unrelated control
sentences = [
    # Pair 1: Cats
    "The cat is sleeping .",
    "The dog is walking .",
]

def get_embeddings(texts):
    """Generate embeddings using OpenAI's text-embedding-3-small model."""
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Extract embeddings from response
    embed= model.encode(texts)    
    return embed

# Get embeddings for all sentences
embeddings = get_embeddings(sentences)

# Compute cosine similarity
similarities = cosine_similarity(embeddings)


print(embeddings)

print("=" * 60)

print("COSINE SIMILARITY MATRIX (10 x 10)")
print(similarities)

# Print results
# print("=" * 60)
# print("COSINE SIMILARITY MATRIX (10 x 10)")
# print("=" * 60)
# print("Sentences 0-9:")
# for i, sent in enumerate(sentences):
#     print(f"{i}: {sent[:50]}...")
# print("\nSimilarity Scores (1.0 = identical, 0.0 = unrelated):")
# print("-" * 60)

# # Define pairs: (index1, index2, label)
# pairs = [
#     (0, 1, "Pair 1: Cat"),
#     (2, 3, "Pair 2: Weather"),
#     (4, 5, "Pair 3: Food"),
#     (6, 7, "Pair 4: Tech"),
#     (8, 9, "Pair 5: Education"),
#     (0, 9, "CONTROL: Cat vs Education")
# ]

# for i, j, label in pairs:
#     score = similarities[i][j]
#     print(f"{label:20} → Score: {score:.4f}")

# # Show full matrix (optional)
# print("\n" + "=" * 60)
# print("FULL SIMILARITY HEATMAP (values rounded to 2 decimals)")
# print("=" * 60)
# np.set_printoptions(precision=2, suppress=True)
# print(similarities.round(2))