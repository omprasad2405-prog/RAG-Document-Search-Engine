# 📚 Document RAG Engine with ChromaDB & FastEmbed (GenAI Lab 3)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR_STREAMLIT_APP_LINK_HERE.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Groq Fast Inference](https://img.shields.io/badge/Groq-Cloud_Inference-orange.svg)](https://console.groq.com/)

A lightweight, production-ready Retrieval-Augmented Generation (RAG) system that extracts text from documents (PDF/TXT), indexes vector embeddings locally using ONNX-based **FastEmbed** and **ChromaDB**, and provides accurate, grounded answers with transparent source chunk citations using **Groq Cloud**.

---

## 🚀 Live Demo

Try the interactive document search application:  
👉 **[Launch Streamlit App](https://rag-document-search-eng.streamlit.app/)**

---

## 📌 Problem Solved

Passing entire multi-page documents (like syllabi, research papers, or manuals) directly into an LLM prompt leads to:
1. **Context Window Saturation:** Quickly overflows token limits on large documents.
2. **High Latency & Costs:** Sending thousands of repetitive tokens slows down generation and increases API costs.
3. **Lost in the Middle:** LLMs frequently hallucinate or overlook specific facts buried inside large text bodies.

This RAG engine solves these challenges by breaking documents into semantic chunks, generating vector embeddings locally via **FastEmbed**, storing them in **ChromaDB**, and retrieving only the top-$K$ most relevant passages to ground the LLM's response.

---

## 🏗️ Architecture

Document Ingestion  ──► Paragraph & Sentence Chunking (~1000 chars, 150 overlap)
│
▼

Vector Embeddings    ──► FastEmbed (BAAI/bge-small-en-v1.5)
│
▼

Local Vector Store   ──► ChromaDB (In-Memory / Local Index)
│
▼

Semantic Retrieval   ──► User Query ──► Cosine Similarity Search (Top-K Chunks)
│
▼

Grounded Synthesis   ──► [System Prompt + Retrieved Chunks + Query]
│
▼
Groq API ──► Cited Answer with Exact Source Chunks

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **LLM Engine:** Groq Cloud API (`openai/gpt-oss-20b` / `llama-3.3-70b-versatile`)
* **Embedding Model:** `FastEmbed` (`BAAI/bge-small-en-v1.5` via ONNX runtime, ~60 MB)
* **Vector Store:** `ChromaDB`
* **PDF Extraction:** `pypdf`
* **Web UI:** Streamlit

---

## 📂 Project Structure

```text
GenAI_lab3/
├── .env                  # API keys (kept secret, ignored by git)
├── .gitignore            # Git exclusion rules
├── requirements.txt      # Lightweight dependencies
├── rag_engine.py         # Core RAG pipeline (Chunking, FastEmbed, ChromaDB, Groq)
├── app.py                # Streamlit web UI & source chunk inspector
└── README.md             # Project documentation