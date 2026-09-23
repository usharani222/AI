# AI Learning & Engineering Lab

[![AI - Antigravity](https://img.shields.io/badge/AI-Antigravity-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-green?style=for-the-badge)](https://www.langchain.com/)
[![LangSmith](https://img.shields.io/badge/Observability-LangSmith-orange?style=for-the-badge)](https://smith.langchain.com/)

A comprehensive hands-on repository of modern generative AI patterns, LLM orchestration with **LangChain**, RAG architectures, streaming, and testing — developed and pair-programmed with **AI - Antigravity** (Google DeepMind).

---

## Featured Project: EventFlow Internal Knowledge Assistant (v1)

Located in [`LangChain/Internal_Knowledge_Assistant v1/`](./LangChain/Internal_Knowledge_Assistant%20v1/):

* **Architecture:** Decoupled LCEL pipeline (`RunnableParallel` ➔ `ToyRetriever` ➔ `ChatPromptTemplate` ➔ `ChatOpenAI` ➔ Tools ➔ Output).
* **Grounded RAG:** Answers questions strictly based on internal EventFlow documentation snippets, explicitly declining out-of-scope inquiries without hallucinating.
* **Tool Calling:** Intercepts system health queries to execute real-time diagnostic checks (`get_service_status`).
* **Resilience & Testing:** Automatic retries, graceful fallbacks, and deterministic unit tests with `pytest`.
* **Full Documentation:** See the [Internal Knowledge Assistant README](./LangChain/Internal_Knowledge_Assistant%20v1/README.md) for zero-context setup and architectural details.