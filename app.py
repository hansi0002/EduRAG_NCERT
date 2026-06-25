import streamlit as st
from rag_engine import NCFTRag
from dotenv import load_dotenv
import os
load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

@st.cache_resource
def load_rag():

    return NCFTRag(API_KEY)


rag = load_rag()

st.set_page_config(
    page_title="NCERT AI Tutor",
    page_icon="📚"
)

st.title("📚 NCERT Class 9 Maths Tutor")

question = st.text_input(
    "Ask a Question"
)

if st.button("Ask"):

    with st.spinner("Searching NCERT..."):

        answer = rag.ask(question)

    st.markdown("### Answer")

    st.write(answer)