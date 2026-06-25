from pypdf import PdfReader
import os
import pickle

os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from sentence_transformers import SentenceTransformer

import faiss
import numpy as np


# -----------------------------
# CHUNKING FUNCTION
# -----------------------------
def create_chunks(text, chunk_size=1000, overlap=200):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# -----------------------------
# LOAD PDFS
# -----------------------------
DATA_FOLDER = "data"

pdf_files = [
    f for f in os.listdir(DATA_FOLDER)
    if f.endswith(".pdf")
]

print(f"Found {len(pdf_files)} PDFs\n")

all_chunks = []

# -----------------------------
# READ EACH PDF
# -----------------------------
for pdf_file in pdf_files:

    pdf_path = os.path.join(DATA_FOLDER, pdf_file)

    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text

    chapter_chunks = create_chunks(text)

    all_chunks.extend(chapter_chunks)

    print("=" * 50)
    print(f"File: {pdf_file}")
    print(f"Pages: {len(reader.pages)}")
    print(f"Characters: {len(text)}")
    print(f"Chunks Created: {len(chapter_chunks)}")
    print("=" * 50)


# -----------------------------
# TOTAL CHUNKS
# -----------------------------
print("\n" + "=" * 50)
print(f"TOTAL CHUNKS CREATED: {len(all_chunks)}")
print("=" * 50)


# -----------------------------
# SAMPLE CHUNK
# -----------------------------
print("\nSample Chunk:\n")
print(all_chunks[0][:500])


# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------
print("\nLoading Embedding Model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

print("Model Loaded!")


# -----------------------------
# GENERATE EMBEDDINGS
# -----------------------------
print("\nGenerating Embeddings...")

embeddings = model.encode(
    all_chunks,
    batch_size=4,
    show_progress_bar=True,
    convert_to_numpy=True
)

embeddings = np.array(
    embeddings,
    dtype=np.float32
)

print("\nEmbedding Shape:", embeddings.shape)


# -----------------------------
# CREATE FAISS INDEX
# -----------------------------
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Vectors Stored:", index.ntotal)


# -----------------------------
# CREATE VECTOR STORE FOLDER
# -----------------------------
os.makedirs("vector_store", exist_ok=True)


# -----------------------------
# SAVE FAISS INDEX
# -----------------------------
faiss.write_index(
    index,
    "vector_store/ncert_index.faiss"
)

print("\nFAISS Index Saved Successfully!")


#------------------------------
#SAVE Chunks
#------------------------------
with open(
    "vector_store/chunks.pkl",
    "wb"
) as f:

    pickle.dump(all_chunks, f)

print("Chunks Saved Successfully!")


# -----------------------------
# SUCCESS MESSAGE
# -----------------------------
print("\nBuild Complete!")