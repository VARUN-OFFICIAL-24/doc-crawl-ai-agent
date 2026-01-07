import asyncio

from crawler.web_crawler import crawl_pydantic_ai_docs
from ingestion.chunking import chunk_documents
from ingestion.embeddings import get_embedding_model, embed_chunks
from vectorstore.chroma_store import (
    get_chroma_store,
    ingest_embeddings,
)


# ==================================================
# Configuration
# ==================================================

START_URL = "https://docs.pydantic.dev/latest/ai/"
MAX_PAGES = 50
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# ==================================================
# Ingestion pipeline
# ==================================================

async def build_vector_store():
    print("🚀 Starting documentation crawl...")

    documents = await crawl_pydantic_ai_docs(
        start_url=START_URL,
        max_pages=MAX_PAGES,
    )

    print(f"✅ Crawled {len(documents)} pages")

    print("✂️ Chunking documents...")
    chunks = chunk_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )

    print(f"✅ Created {len(chunks)} text chunks")

    print("🧠 Generating embeddings...")
    embedding_model = get_embedding_model()
    embedded_chunks = embed_chunks(chunks, embedding_model)

    print("✅ Embeddings generated")

    print("📦 Storing embeddings in ChromaDB...")
    vectorstore = get_chroma_store(
        persist_dir="data/chroma",
        embedding_model=embedding_model,
    )
    ingest_embeddings(embedded_chunks, vectorstore)

    print("🎉 Vector store built successfully!")
    print("📂 Stored locally at: data/chroma/")


# ==================================================
# Entry point
# ==================================================

if __name__ == "__main__":
    asyncio.run(build_vector_store())

