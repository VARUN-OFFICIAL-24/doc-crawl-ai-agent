
import streamlit as st

from ingestion.embeddings import get_embedding_model
from vectorstore.chroma_store import (
    get_chroma_store,
    similarity_search,
)
from agent.doc_agent import get_llm, answer_question


# ==================================================
# Streamlit page setup
# ==================================================

st.set_page_config(
    page_title="Documentation AI Assistant",
    layout="wide",
)

st.title("📚 Documentation AI Assistant")
st.write(
    "Ask questions over crawled technical documentation using an agentic RAG system."
)

# ==================================================
# Load models and vector store (cached)
# ==================================================

@st.cache_resource
def load_resources():
    embedding_model = get_embedding_model()
    vectorstore = get_chroma_store(
        persist_dir="data/chroma",
        embedding_model=embedding_model,
    )
    llm = get_llm()
    return vectorstore, llm


vectorstore, llm = load_resources()

# ==================================================
# User input
# ==================================================

query = st.text_input(
    "Enter your question:",
    placeholder="e.g. How does Pydantic AI handle tools?",
)

if query:
    with st.spinner("Searching documentation..."):
        docs = similarity_search(
            query=query,
            vectorstore=vectorstore,
            k=5,
        )

    if not docs:
        st.warning("No relevant documentation found.")
    else:
        with st.spinner("Generating answer..."):
            answer = answer_question(
                query=query,
                retrieved_docs=docs,
                llm=llm,
            )

        st.subheader("Answer")
        st.write(answer)

        with st.expander("Sources"):
            for i, doc in enumerate(docs, start=1):
                st.markdown(
                    f"**Source {i}**  \n{doc.metadata.get('url', 'Unknown')}"
                )
