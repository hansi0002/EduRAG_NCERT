from sentence_transformers import SentenceTransformer
import faiss
import pickle
import numpy as np


# -----------------------------
# LOAD FAISS INDEX
# -----------------------------
index = faiss.read_index(
    "vector_store/ncert_index.faiss"
)

print("FAISS Index Loaded!")


# -----------------------------
# LOAD CHUNKS
# -----------------------------
with open(
    "vector_store/chunks.pkl",
    "rb"
) as f:

    chunks = pickle.load(f)

print("Chunks Loaded!")


# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding Model Loaded!")


# -----------------------------
# USER QUERY
# -----------------------------
query = input("\nAsk a Question: ")


# -----------------------------
# QUERY EMBEDDING
# -----------------------------
query_embedding = model.encode(query)

query_embedding = np.array(
    [query_embedding],
    dtype=np.float32
)


# -----------------------------
# SEARCH TOP 3 CHUNKS
# -----------------------------
k = 5

distances, indices = index.search(
    query_embedding,
    k
)


# -----------------------------
# DISPLAY RESULTS
# -----------------------------
print("\nTop Results:\n")

for i, idx in enumerate(indices[0]):

    print("=" * 70)

    print(f"Result {i+1}")
    print(f"Chunk Index: {idx}")
    print(f"Distance: {distances[0][i]}")

    print("\nChunk Text:\n")

    print(chunks[idx][:500])

    print("\n")