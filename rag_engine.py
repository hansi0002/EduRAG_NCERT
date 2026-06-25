import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
import google.generativeai as genai


class NCFTRag:

    def __init__(self, api_key):

        # Gemini
        genai.configure(api_key=api_key)

        self.llm = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

        # FAISS
        self.index = faiss.read_index(
            "vector_store/ncert_index.faiss"
        )

        print("FAISS Loaded!")

        # Chunks
        with open(
            "vector_store/chunks.pkl",
            "rb"
        ) as f:

            self.chunks = pickle.load(f)

        print("Chunks Loaded!")

        # Embedding Model
        self.embed_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Embedding Model Loaded!")

    def ask(self, question, history=""):

        # Query Embedding
        query_embedding = self.embed_model.encode(
            question
        )

        query_embedding = np.array(
            [query_embedding],
            dtype=np.float32
        )

        # Retrieval
        distances, indices = self.index.search(
            query_embedding,
            3
        )

        # Context Building
        context = ""

        for idx in indices[0]:
            context += self.chunks[idx]
            context += "\n\n"

        # Prompt
        prompt = f"""
You are an expert NCERT Class 9 Mathematics Teacher.

Instructions:
- Answer only from the provided context.
- Use conversation history when relevant.
- Explain in simple student-friendly language.
- Give examples whenever possible.
- If the answer is not present in the context, say:
  "I could not find this in the NCERT knowledge base."

Conversation History:
{history}

Context:
{context}

Current Question:
{question}
"""

        # Gemini Response
        response = self.llm.generate_content(
            prompt
        )

        return response.text