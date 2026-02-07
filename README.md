# Acadamic-RAG-System(Local, PDF-based)

This repository contains a minimal yet complete Retrieval-Augmented Generation (RAG) system
designed for academic documents such as research papers, lecture PDFs, and study materials.

The system performs:
- PDF ingestion
- Text chunking
- Dense embedding using Sentence Transformers
- Vector search using FAISS
- Grounded answer generation using a local LLM (Ollama + LLaMA 3)

---

## Features

- Fully local pipeline (no paid APIs)
- Designed for academic PDFs
- Explainable and modular architecture
- Suitable for exam preparation and research assistance

## Project Structure

RAG_System/
├── src/
│ ├── ingest.py # PDF ingestion
│ ├── chunking.py # Token-aware chunking
│ ├── embed.py # Embedding generation
│ ├── index.py # FAISS index creation
│ ├── retrieve.py # Semantic retrieval
│ └── rag.py # End-to-end RAG pipeline
│
├── data/
│ ├── raw/ # Input PDFs
│ └── artifacts/ # Generated data (ignored in git)
│
├── requirements.txt
├── .gitignore
└── README.md





### Setup Instructions

## Setup Instructions

### 1. Create virtual environment

python -m venv .venv
Windows: .venv\Scripts\activate

### 2. Install dependencies

pip install -r requirements.txt

### 3. Install and Run Ollama

(Note: We has used model llama3:8b here, the model selection depends on your system specifications)
ollama pull llama3:8b
ollama serve

## Running the pipeline

Step 1: Ingest PDFs

Place PDFs in data/raw/ and run:

python src/ingest.py

Step 2: Chunk documents

python src/chunking.py

Step 3: Create embeddings

python src/embed.py

Step 4: Build FAISS index

python src/index.py

Step 5: Ask Queries

python src/rag.py






### Limitations

- Retrieval ranking may favor experiment-heavy sections

- No hybrid (BM25 + dense) search yet

- Designed for small-to-medium document collections
