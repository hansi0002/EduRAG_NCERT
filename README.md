# EduRAG_NCERT
# NCERT AI Tutor (RAG)

An AI-powered tutor for NCERT Class 9 Mathematics built using:

- Gemini
- FAISS
- Sentence Transformers
- Streamlit

## Features

- Ask questions from NCERT textbooks
- Retrieval-Augmented Generation (RAG)
- Semantic Search
- Streamlit UI

## Setup

pip install -r requirements.txt

Create .env

GOOGLE_API_KEY=your_api_key

Run:

python build_index.py

streamlit run app.py
