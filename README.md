# Doc-Crawl AI Agent

Doc-Crawl AI Agent is an **agentic documentation intelligence system** that crawls technical documentation, builds a local vector database, and enables grounded question answering using retrieval-augmented generation (RAG).

The project demonstrates how **web crawling, embeddings, vector search, and LLM reasoning** can be combined into a clean, end-to-end AI system.

## 🔍 What This Project Does

- ✅ Crawls real documentation websites
- ✅ Extracts and chunks structured content
- ✅ Generates embeddings locally using Ollama
- ✅ Stores vectors in a persistent ChromaDB index
- ✅ Answers user questions using **only retrieved documentation context**
- ✅ Provides a simple Streamlit-based query interface

**This is not a PDF chatbot or a static demo** — it is a **reusable documentation intelligence pipeline**.

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
