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

## 🧠 System Flow (High Level)

**Crawl → Chunk → Embed → Store → Retrieve → Reason → Answer**


## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Pull required Ollama models
```bash
ollama pull llama3.2
ollama pull nomic-embed-text

```
---

### 📦 Build the Vector Store (One-Time)

Run the ingestion pipeline locally:
```bash
python src/main.py

```
This creates a local vector database at data/chroma/
(Generated data is intentionally not committed to GitHub.)


### 💬 Query the Documentation

Start the Streamlit interface:

```bash
streamlit run src/ui/streamlit_app.py

```
Ask natural-language questions over the crawled documentation.

---

### 📁 Repository Notes

- Vector data is generated locally and ignored via .gitignore
- The repository contains code only, not scraped content
- The system is designed to be extended to other documentation sources

---

### 🎯 Use Cases

- 📚 Documentation search and Q&A
- 🛠️ Developer support tools
- 💼 Internal knowledge bases
- 🏗️ Reference architecture for agentic RAG systems

---

## ⚠️ Disclaimer

`This project is intended for educational and experimental use.
Always verify answers against official documentation.`



**Built as a clean, local-first agentic RAG system using modern Python tooling.**
