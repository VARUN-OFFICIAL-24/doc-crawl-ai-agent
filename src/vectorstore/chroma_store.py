from typing import List, Dict
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


# ==================================================
# Vector store setup
# ==================================================

def get_chroma_store(
    persist_dir: str = "data/chroma",
    embedding_model: OllamaEmbeddings | None = None,
) -> Chroma:
    """
    Initialize or load a Chroma vector store.

    Args:
        persist_dir: Directory to persist vectors
        embedding_model: Embedding model instance

    Returns:
        Chroma vector store
    """
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    if embedding_model is None:
        embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model,
    )


# ==================================================
# Ingestion into vector store
# ==================================================

def ingest_embeddings(
    embedded_chunks: List[Dict],
    vectorstore: Chroma,
) -> None:
    """
    Store embedded text chunks in Chroma.

    Expected input format:
    {
        "url": "...",
        "chunk_id": 0,
        "text": "...",
        "embedding": [...]
    }
    """
    documents = []
    metadatas = []
    ids = []

    for item in embedded_chunks:
        doc_id = f"{item['url']}::chunk_{item['chunk_id']}"

        documents.append(item["text"])
        metadatas.append(
            {
                "url": item["url"],
                "chunk_id": item["chunk_id"],
            }
        )
        ids.append(doc_id)

    vectorstore.add_texts(
        texts=documents,
        metadatas=metadatas,
        ids=ids,
    )

    vectorstore.persist()


# ==================================================
# Similarity search
# ==================================================

def similarity_search(
    query: str,
    vectorstore: Chroma,
    k: int = 5,
) -> List[Document]:
    """
    Perform similarity search over the vector store.

    Args:
        query: Natural language query
        vectorstore: Chroma store
        k: Number of results

    Returns:
        List of LangChain Document objects
    """
    return vectorstore.similarity_search(query, k=k)

