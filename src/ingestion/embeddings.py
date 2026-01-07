from typing import List, Dict
from langchain_ollama import OllamaEmbeddings


# ==================================================
# Embedding model setup
# ==================================================

def get_embedding_model(
    model_name: str = "nomic-embed-text",
) -> OllamaEmbeddings:
    """
    Initialize and return an Ollama embedding model.

    Args:
        model_name: Name of the embedding model pulled via Ollama

    Returns:
        OllamaEmbeddings instance
    """
    return OllamaEmbeddings(model=model_name)


# ==================================================
# Embedding generation
# ==================================================

def embed_chunks(
    chunks: List[Dict],
    embedding_model: OllamaEmbeddings,
) -> List[Dict]:
    """
    Generate embeddings for text chunks.

    Expected input format:
    {
        "url": "...",
        "chunk_id": 0,
        "text": "..."
    }

    Output format:
    {
        "url": "...",
        "chunk_id": 0,
        "text": "...",
        "embedding": [...]
    }
    """
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.embed_documents(texts)

    embedded_chunks = []
    for chunk, embedding in zip(chunks, embeddings):
        embedded_chunks.append(
            {
                "url": chunk["url"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "embedding": embedding,
            }
        )

    return embedded_chunks
