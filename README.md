# 🤖 LLM Local Assistant

A locally-running AI assistant that answers questions about restaurant reviews using Retrieval-Augmented Generation (RAG). Built with LangChain, ChromaDB, and Ollama — no cloud APIs, no data leaving your machine.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

---

## Overview

This project demonstrates a fully local RAG pipeline where:
1. Restaurant review data is loaded from a CSV file
2. Reviews are chunked, embedded, and stored in a local ChromaDB vector store
3. A user query is matched against the most relevant reviews
4. A local LLM (via Ollama) generates a grounded answer based on retrieved context

No internet connection or paid API required after initial model setup.

---

## Features

- ✅ Fully local — no data sent to external APIs
- ✅ RAG pipeline using LangChain + ChromaDB
- ✅ Persistent vector store (no re-embedding on every run)
- ✅ Query restaurant reviews in natural language
- ✅ Runs on consumer hardware via Ollama

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM Runtime | [Ollama](https://ollama.com) |
| LLM Model | llama3 / mistral (configurable) |
| Embeddings | nomic-embed-text (via Ollama) |
| Vector Store | ChromaDB |
| Orchestration | LangChain |
| Language | Python 3.10+ |
| Data | CSV (restaurant reviews) |

---

## Project Structure

```
LLM_Local/
├── main.py                        # Main entrypoint — query the assistant
├── vector.py                      # Embeds data and builds ChromaDB vector store
├── realistic_restaurant_reviews.csv  # Source data
├── requirements.txt               # Python dependencies
├── .gitignore                     # Ignored files
└── README.md                      # This file
```

---

## Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.10 or higher
- [Ollama](https://ollama.com/download) installed and running
- Git

### Pull required Ollama models

```bash
# Pull the LLM
ollama pull llama3

# Pull the embedding model
ollama pull nomic-embed-text
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ebhavaniprasad99/LLM_Local.git
cd LLM_Local
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Build the vector store

Run this once to embed the restaurant reviews into ChromaDB:

```bash
python vector.py
```

You should see a `chroma_langchain_db/` folder created locally.

---

## Usage

Start the assistant and ask questions about the restaurant reviews:

```bash
python main.py
```

### Example queries

```
> What restaurants have the best pasta?
> Which places are good for a family dinner?
> Are there any restaurants with outdoor seating?
> What do customers say about service quality?
```

Type `exit` or `quit` to stop the assistant.

---

## How It Works

```
User Query
    │
    ▼
Embed Query (nomic-embed-text)
    │
    ▼
ChromaDB Vector Search
(finds top-k most relevant reviews)
    │
    ▼
Build Prompt with Context
    │
    ▼
Local LLM via Ollama (llama3)
    │
    ▼
Answer
```

1. **Ingestion** (`vector.py`): The CSV is loaded, split into chunks, and each chunk is embedded using `nomic-embed-text`. Embeddings are stored in ChromaDB.
2. **Retrieval** (`main.py`): Your question is embedded and compared against stored vectors. The top matching review chunks are retrieved.
3. **Generation** (`main.py`): The retrieved chunks are injected into a prompt and sent to the local LLM, which generates a grounded answer.

---

## Configuration

Key settings are currently defined inside the Python files. Future versions will move these to a `.env` file.

| Setting | Location | Default | Description |
|---|---|---|---|
| LLM model name | `main.py` | `llama3` | Ollama model to use for generation |
| Embedding model | `vector.py` | `nomic-embed-text` | Ollama model for embeddings |
| Top-k retrieval | `main.py` | `5` | Number of chunks to retrieve per query |
| Chunk size | `vector.py` | `500` | Characters per chunk |
| Chunk overlap | `vector.py` | `50` | Overlap between chunks |

---

## Known Limitations

- No conversation memory — each question is independent (no follow-up support)
- Single data source — only restaurant reviews CSV supported
- No web UI — command line only
- No error handling for missing Ollama models
- Configuration is hardcoded, not environment-variable driven

---

## Roadmap

- [ ] Add `.env` support for configuration
- [ ] Add multi-turn conversation memory
- [ ] Add Streamlit web UI
- [ ] Support PDF and additional file formats
- [ ] Add FastAPI REST endpoint
- [ ] Add Docker support
- [ ] Add unit tests and RAG evaluation metrics
- [ ] Hybrid search (vector + keyword)

---

## Contributing

Contributions are welcome! To get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "feat: add your feature"`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---
