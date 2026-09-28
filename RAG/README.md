# Advanced RAG System & Engineering Lab

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-green.svg)](https://www.langchain.com/)
[![Chroma](https://img.shields.io/badge/Vector%20DB-Chroma-red.svg)](https://www.trychroma.com/)
[![Embeddings](https://img.shields.io/badge/Embeddings-MiniLM--L6--v2-yellow.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

This directory contains a complete, production-grade **Retrieval-Augmented Generation (RAG)** pipeline covering the entire engineering lifecycle: chunking, embedding, metadata filtering, citations, hybrid retrieval, automated evaluation, and production hardening.

---

## 📂 Architecture & File Map

| Script / File | Purpose & Concepts Covered |
|---|---|
| [`pipeline.py`](./pipeline.py) | **Baseline RAG Pipeline**: PDF text loading, chunking, Chroma ingestion, and grounded QA chain. |
| [`chunking_comparison.py`](./chunking_comparison.py) | **Chunking Strategies**: Evaluating chunk sizes, overlap settings, and separator boundaries. |
| [`embeddings.py`](./embeddings.py) | **Vector Representations**: Generating and comparing dense embeddings via `all-MiniLM-L6-v2`. |
| [`metadata.py`](./metadata.py) | **Metadata Filtering & Security Boundaries**: Hard retrieval-layer access control preventing data leakage across teams. |
| [`citations.py`](./citations.py) | **Source Grounding & Provenance**: Accurate in-text citations (`[Source, Page]`) and hallucination refusal. |
| [`hybrid_search.py`](./hybrid_search.py) | **Hybrid Search**: BM25 sparse keyword matching + Dense semantic vectors fused via **Reciprocal Rank Fusion (RRF)**. |
| [`eval.py`](./eval.py) | **Evaluation Framework**: Automated citation subset testing, human-in-the-loop review, and pass-rate tracking. |
| [`eval_set.json`](./eval_set.json) | **Ground Truth Benchmark**: Curated questions, reference answers, and expected PDF sources. |
| [`eval_results.json`](./eval_results.json) | **Evaluation Results**: Saved metrics, answer correctness rates, and citation verification logs. |
| [`prod_rag.py`](./prod_rag.py) | **Production RAG Architecture**: Tenant authorization, SHA-256 hashed response caching, and structured logging. |
| [`data/pdf/`](./data/pdf/) | **Knowledge Corpus**: 10 technical reference PDFs on HTTP, REST APIs, TLS, auth, and idempotency. |

---

## 🚀 Key Modules & How to Run

### 1. Hybrid Search (BM25 + Semantic + RRF)
Combines the exact-match precision of BM25 (status codes, headers, identifiers) with the paraphrasing power of dense vector embeddings:

```powershell
python hybrid_search.py
```

* **Formula**:
  $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{60 + \text{rank}_m(d)}$$

---

### 2. RAG Evaluation Benchmark (`eval.py`)
Evaluates the RAG system across both **Answer Correctness** and **Citation Faithfulness**:

```powershell
python eval.py
```

* **Workflow**:
  1. Loads questions and ground truth from `eval_set.json`.
  2. Retrieves chunks and tags them with labels `[S1]`, `[S2]`.
  3. Prompts the LLM to answer strictly citing those labels.
  4. Automatically tests if the cited sources contain all expected sources (`expected ⊆ cited`).
  5. Solicits human validation and exports summary metrics to `eval_results.json`.

---

### 3. Production RAG (`prod_rag.py`)
Demonstrates production engineering patterns:

```powershell
python prod_rag.py
```

* **Pre-Retrieval Authorization**: Blocks unauthorized users before executing queries.
* **Tenant-Scoped Caching**: Fast cache hits for repeated queries using `SHA-256(user_id|tenant_id|normalized_query)`.
* **Outage Fallbacks**: Catches database/LLM failures gracefully without crashing.

---

## 📦 Prerequisites & Dependencies

Make sure your dependencies are installed from `requirements.txt`:

```powershell
pip install -r requirements.txt
```

Set your API key in `.env`:
```env
OPENROUTER_API_KEY=your_key_here
```
