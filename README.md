# AI Learning & Engineering Lab

[![AI - Antigravity](https://img.shields.io/badge/AI-Antigravity-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-green?style=for-the-badge)](https://www.langchain.com/)
[![Chroma](https://img.shields.io/badge/Vector%20DB-Chroma-red?style=for-the-badge)](https://www.trychroma.com/)
[![HuggingFace](https://img.shields.io/badge/Embeddings-HuggingFace-yellow?style=for-the-badge)](https://huggingface.co/)
[![LangSmith](https://img.shields.io/badge/Observability-LangSmith-orange?style=for-the-badge)](https://smith.langchain.com/)

A comprehensive hands-on repository of modern generative AI patterns, LLM orchestration with **LangChain**, advanced RAG architectures, hybrid retrieval, automated evaluation, and production hardening — developed and pair-programmed with **AI - Antigravity** (Google DeepMind).

---

## 📂 Repository Structure

```
AI/
├── LangChain/                             # LangChain foundations, chains & tools
│   └── Internal_Knowledge_Assistant v1/   # Featured event booking assistant
├── RAG/                                   # Advanced RAG System & Engineering
│   ├── data/pdf/                          # Corpus of technical reference PDFs
│   ├── pipeline.py                        # Baseline PDF/Text ingestion & RAG chain
│   ├── chunking_comparison.py             # Text splitting & chunk size benchmarks
│   ├── embeddings.py                      # Dense vector representation with all-MiniLM-L6-v2
│   ├── metadata.py                        # Retrieval-layer metadata filtering & ACLs
│   ├── citations.py                       # Grounded generation with provenance citations
│   ├── hybrid_search.py                   # BM25 + Dense Vectors + Reciprocal Rank Fusion (RRF)
│   ├── eval.py                            # Automated benchmark evaluation & citation verification
│   ├── eval_set.json                      # Ground-truth evaluation dataset
│   ├── eval_results.json                  # Benchmark pass rates & evaluation logs
│   └── prod_rag.py                        # Production-ready RAG with auth, caching & resilience
└── README.md
```

---

## 🚀 Featured Project 1: Advanced RAG System & Production Architecture

Located in [`RAG/`](./RAG/):

### Key Capabilities

1. **Hybrid Retrieval (BM25 + Dense Semantic + RRF)**:
   - Fuses **BM25Okapi** sparse keyword matching with **sentence-transformers** dense vectors.
   - Merges results via **Reciprocal Rank Fusion (RRF)**:
     $$\text{RRF\_Score}(d) = \sum_{m} \frac{1}{k + \text{rank}_m(d)}$$
   - Solves both exact keyword queries (`429`, `Idempotency-Key`) and conceptual paraphrased queries.

2. **Retrieval-Layer Metadata Filtering & Access Control**:
   - Implements hard retrieval boundaries in Chroma.
   - Unauthorized tenant documents are never fetched, preventing prompt leakage or model hallucinations.

3. **Citations & Provenance Grounding**:
   - Strictly enforces in-text bracketed citations (`[S1]`, `[S2]`) with source file and page mapping.
   - Anti-hallucination fallback: Explicitly declines when context is insufficient.

4. **Automated Evaluation Benchmark (`eval.py`)**:
   - Automated citation subset verification against ground truth (`eval_set.json`).
   - Human-in-the-loop review for answer correctness and citation faithfulness.
   - Comprehensive metric reporting (`eval_results.json`) tracking pass rates.

5. **Production Hardening (`prod_rag.py`)**:
   - **Pre-Retrieval Authorization**: Multi-tenant validation before query execution.
   - **Scoped Response Caching**: SHA-256 hashed query cache scoped by `(user_id, tenant_id)`.
   - **Fault Tolerance**: Graceful degradations and structured audit logging.

---

## 🛠️ Featured Project 2: EventFlow Internal Knowledge Assistant (v1)

Located in [`LangChain/Internal_Knowledge_Assistant v1/`](./LangChain/Internal_Knowledge_Assistant%20v1/):

* **Architecture:** Decoupled LCEL pipeline (`RunnableParallel` ➔ `ToyRetriever` ➔ `ChatPromptTemplate` ➔ `ChatOpenAI` ➔ Tools ➔ Output).
* **Grounded RAG:** Answers questions strictly based on internal EventFlow documentation snippets, explicitly declining out-of-scope inquiries without hallucinating.
* **Tool Calling:** Intercepts system health queries to execute real-time diagnostic checks (`get_service_status`).
* **Resilience & Testing:** Automatic retries, graceful fallbacks, and deterministic unit tests with `pytest`.
* **Full Documentation:** See the [Internal Knowledge Assistant README](./LangChain/Internal_Knowledge_Assistant%20v1/README.md) for zero-context setup and architectural details.

---

## ⚡ Quick Start

### 1. Environment Setup

```powershell
# Clone the repository and navigate to root
cd C:\Users\usha7\OneDrive\Desktop\USHA-AI\AI

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r RAG/requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the workspace root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 3. Run RAG Modules

```powershell
# Run Hybrid Search (BM25 + Semantic + RRF)
python RAG/hybrid_search.py

# Run RAG Evaluation Benchmark
python RAG/eval.py

# Run Production RAG Pipeline (with caching and auth)
python RAG/prod_rag.py
```