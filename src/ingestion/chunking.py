from typing import List, Dict


# ==================================================
# Text chunking utilities
# ==================================================

def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Raw text to split
        chunk_size: Number of characters per chunk
        overlap: Number of overlapping characters between chunks

    Returns:
        List of text chunks
    """
    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())

        start = end - overlap

    return chunks


# ==================================================
# Document-level chunking
# ==================================================

def chunk_documents(
    documents: List[Dict],
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[Dict]:
    """
    Chunk a list of crawled documents.

    Each output chunk keeps metadata for traceability.

    Expected input format:
    {
        "url": "...",
        "content": "..."
    }

    Returns:
    [
        {
            "url": "...",
            "chunk_id": 0,
            "text": "..."
        },
        ...
    ]
    """
    all_chunks = []

    for doc in documents:
        url = doc.get("url")
        content = doc.get("content", "")

        chunks = chunk_text(
            text=content,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for idx, chunk in enumerate(chunks):
            all_chunks.append(
                {
                    "url": url,
                    "chunk_id": idx,
                    "text": chunk,
                }
            )

    return all_chunks
