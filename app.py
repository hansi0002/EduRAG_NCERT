import streamlit as st
from rag_engine import NCFTRag
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")


#@st.cache_resource
def load_rag():
    return NCFTRag(API_KEY)


rag = load_rag()

st.set_page_config(
    page_title="NCERT AI Tutor",
    page_icon="📚"
)

st.title("📚 NCERT Class 9 Maths Tutor")


# -------------------------
# CHAT MEMORY
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# -------------------------
# DISPLAY OLD MESSAGES
# -------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -------------------------
# CHAT INPUT
# -------------------------
question = st.chat_input(
    "Ask your NCERT question..."
)


if question:

    # Show user message
    with st.chat_message("user"):
        st.markdown(question)

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Build conversation history
    history = ""

    for msg in st.session_state.messages[-6:]:

        history += (
            f"{msg['role']}: "
            f"{msg['content']}\n"
        )

    # Generate answer
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer, rewritten_query = rag.ask(
                question,
                history
            )
        st.caption(f"Rewritten Query: {rewritten_query}")

        st.markdown(answer)

    # Save assistant message
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )