Backend Docs Mini-RAG Dataset

This folder contains 10 short PDF documents about HTTP and backend API fundamentals.
Use them as a small, coherent corpus for RAG practice.

Suggested workflow:
1. Load each PDF with PyPDFLoader.
2. Preserve source and page metadata.
3. Split loaded Document objects with split_documents().
4. Add a chunk_index to each chunk's metadata.
5. Embed and store chunks in Chroma.
6. Query across the entire collection and return citations from metadata.

Suggested test questions:
1. What does HTTP statelessness mean?
2. What is the difference between PUT and PATCH?
3. What does status code 403 mean?
4. What does the Content-Type header specify?
5. What security properties does HTTPS provide?
6. What is the difference between authentication and authorization?
7. Why can retrying a timed-out payment request be risky?
8. What is an idempotency key?
9. What should an API contract document?
10. What is the difference between 4xx and 5xx errors?
11. What color is the API dashboard? (Not specified; assistant should abstain.)

All documents are intentionally concise and contain answerable facts for testing retrieval, grounding, and citations.
