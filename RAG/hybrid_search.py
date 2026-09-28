"""

Why Hybrid Search?
- BM25 (Sparse Keyword Search):
    * Excels at exact term matching (product IDs, status codes like '429', 
      header names like 'Idempotency-Key', acronyms, and unique technical terms).
    * Fails at vocabulary mismatch (synonyms, conceptual paraphrasing).
- Semantic Search (Dense Vector Search):
    * Excels at understanding meaning, intent, and synonyms (e.g., mapping
      "prevent double charges" to "idempotent API requests").
    * Fails at exact alphanumeric codes, rare technical identifiers, or negation.
- Hybrid Search with Reciprocal Rank Fusion (RRF):
    * Combines the strengths of both methods to produce a robust, unified ranking.
================================================================================
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles to prevent encoding errors
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Document loading & splitting
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Dense vector embeddings & vector store
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Sparse keyword search algorithm
from rank_bm25 import BM25Okapi


# ------------------------------------------------------------------------------
# 1. Load Documents
# ------------------------------------------------------------------------------
# Directory containing our knowledge base PDF documents
DATA_DIR = "./data/pdf"

# DirectoryLoader scans the target folder recursively for all .pdf files
# PyPDFLoader parses each PDF page into a LangChain Document object
loader = DirectoryLoader(
    path=DATA_DIR,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True
)

# documents is a list of Document objects (one per page), each having:
# - page_content: raw extracted text
# - metadata: {'source': '...', 'page': 0}
documents = loader.load()

# Guard: Ensure documents were found before proceeding
if not documents:
    raise ValueError(f"No PDF documents found in {DATA_DIR}")

print(f"Loaded {len(documents)} pages from {DATA_DIR}")


# ------------------------------------------------------------------------------
# 2. Split Documents into Manageable Chunks
# ------------------------------------------------------------------------------
# Large documents exceed embedding model context windows and dilute search precision.
# We break text into coherent ~500-character segments.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,       # Target character length per chunk
    chunk_overlap=50,     # Overlap prevents cutting a critical sentence across boundaries
    separators=["\n\n", "\n", " ", ""], # Priority list of split points to preserve structure
    length_function=len
)

chunks = splitter.split_documents(documents)

# Assign a sequential, deterministic chunk_index to track chunk positions
for index, chunk in enumerate(chunks):
    chunk.metadata["chunk_index"] = index

print(f"Created {len(chunks)} chunks across all documents")


# ------------------------------------------------------------------------------
# 3. Create Semantic Search Index (Dense Vector Search with Chroma)
# ------------------------------------------------------------------------------
# sentence-transformers/all-MiniLM-L6-v2 maps text chunks into 384-dimensional dense vectors
# Vectors capture semantic meaning and conceptual similarity.
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Persistent Chroma vector store saves embeddings to disk
vector_store = Chroma(
    collection_name="hybrid_search_demo",
    embedding_function=embeddings_model,
    persist_directory="./hybrid_chroma_db"
)

# Avoid duplicate chunk insertions when re-running the script
# Only insert chunks if the collection is currently empty
if vector_store._collection.count() == 0:
    # Use deterministic chunk IDs to avoid duplicate insertions
    chunk_ids = [
        f"{c.metadata.get('source', 'doc')}_{c.metadata.get('page', 0)}_{i}"
        for i, c in enumerate(chunks)
    ]
    vector_store.add_documents(chunks, ids=chunk_ids)
    print(f"Ingested {len(chunks)} chunks into Chroma vector store.")
else:
    print(f"Chroma collection already contains {vector_store._collection.count()} chunks. Skipping re-indexing.")


# ------------------------------------------------------------------------------
# 4. Create BM25 Keyword Search Index (Sparse Term Matching)
# ------------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """
    Basic tokenizer for BM25 keyword matching.
    Lowercases text and splits on whitespace so searches are case-insensitive.
    """
    return text.lower().split()


# Prepare tokenized corpus: BM25 expects a list of token lists (e.g. [['http', 'methods'], ...])
tokenized_chunks = [
    tokenize(chunk.page_content)
    for chunk in chunks
]

# Initialize BM25Okapi:
# - Computes Term Frequency (TF) with saturation
# - Computes Inverse Document Frequency (IDF) to give high weight to rare, specific keywords
# - Normalizes by chunk length so longer chunks don't unfairly dominate
bm25 = BM25Okapi(tokenized_chunks)
print("BM25 sparse keyword index created successfully.")


# ------------------------------------------------------------------------------
# 5. Generate a Stable Key for Deduplication
# ------------------------------------------------------------------------------
def document_key(doc):
    """
    Generates a unique, hashable identifier for each chunk.
    
    Why this is needed:
    BM25 and Chroma retrieve separate Python object instances of the same chunk.
    This tuple fingerprint (source file, page number, chunk index, content)
    allows Reciprocal Rank Fusion to identify when both systems retrieved the same document.
    """
    return (
        doc.metadata.get("source", "unknown"),
        doc.metadata.get("page", "unknown"),
        doc.metadata.get("chunk_index", "unknown"),
        doc.page_content
    )


# ------------------------------------------------------------------------------
# 6. Reciprocal Rank Fusion (RRF)
# ------------------------------------------------------------------------------
def reciprocal_rank_fusion(
    result_lists: list[list],
    rrf_k: int = 60,
    top_k: int = 5
):
    """
    Combines multiple ranked lists into a single consensus ranking.
    
    Mathematical Formula:
        RRF_Score(d) = SUM_{m in retrieval_methods} [ 1 / (rrf_k + rank_m(d)) ]
    
    Why RRF is the gold standard for Hybrid Search:
    1. Score Incompatibility Problem:
       - BM25 scores are unbounded positive numbers (e.g. 0 to 25+).
       - Vector cosine similarity scores range from -1.0 to 1.0 (or distances 0 to 2).
       - You cannot simply add or average these raw scores without complex normalization.
    2. Rank-Based Solution:
       - RRF operates purely on RANKS (position 1, position 2, ...), not raw scores.
       - A document ranked #1 gets 1 / (60 + 1) = 0.01639.
       - A document ranked #2 gets 1 / (60 + 2) = 0.01612.
    3. Consensus Bonus:
       - A document appearing near the top of BOTH lists receives a high combined score,
         surpassing documents that only scored well in one method.
    4. rrf_k Smoothing Constant (default = 60):
       - Prevents top ranks from excessively outscoring slightly lower ranks,
         ensuring balanced, stable fusion.
    """
    fused_scores = {}
    documents_by_key = {}

    # Iterate over each retriever's ranked results (BM25 list, then Semantic list)
    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            key = document_key(doc)

            # Accumulate the reciprocal rank score for this document
            fused_scores[key] = (
                fused_scores.get(key, 0.0)
                + 1 / (rrf_k + rank)
            )

            # Store reference to the Document object
            documents_by_key[key] = doc

    # Sort documents by their combined RRF score in descending order
    ranked_keys = sorted(
        fused_scores,
        key=fused_scores.get,
        reverse=True
    )

    # Return top_k documents as (Document, rrf_score) pairs
    return [
        (documents_by_key[key], fused_scores[key])
        for key in ranked_keys[:top_k]
    ]


# ------------------------------------------------------------------------------
# 7. Hybrid Search Coordinator
# ------------------------------------------------------------------------------
def hybrid_search(query: str, top_k: int = 5):
    """
    Orchestrates Hybrid Search in 3 steps:
    1. Runs BM25 keyword search across all chunks.
    2. Runs Chroma dense semantic search across vector embeddings.
    3. Fuses both ranked lists using Reciprocal Rank Fusion (RRF).
    
    Returns:
        (bm25_results, semantic_results, hybrid_results)
    """
    # --- Step A: BM25 Sparse Keyword Search ---
    # Tokenize the user's query with the same tokenizer used for chunks
    query_tokens = tokenize(query)
    
    # Calculate BM25 relevance score for each chunk in the corpus
    bm25_scores = bm25.get_scores(query_tokens)

    # Get the indices of the top_k highest scoring chunks
    bm25_ranked_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:top_k]

    # Map indices back to Document objects
    bm25_results = [
        chunks[index]
        for index in bm25_ranked_indices
    ]

    # --- Step B: Dense Semantic Vector Search ---
    # Chroma embeds the query vector and finds the top_k nearest chunk vectors
    semantic_results = vector_store.similarity_search(
        query=query,
        k=top_k
    )

    # --- Step C: Reciprocal Rank Fusion ---
    # Merge both ranked result sets into a single, unified ranking
    hybrid_results = reciprocal_rank_fusion(
        result_lists=[
            bm25_results,
            semantic_results
        ],
        rrf_k=60,
        top_k=top_k
    )

    return bm25_results, semantic_results, hybrid_results


# ------------------------------------------------------------------------------
# 8. Display Search Results
# ------------------------------------------------------------------------------
def display_results(title: str, results):
    """
    Prints search results formatted clearly with rank, metadata, score, and content snippet.
    """
    print(f"\n{'=' * 75}")
    print(f"📋 {title}")
    print("=" * 75)

    if not results:
        print("No results returned.")
        return

    for rank, item in enumerate(results, start=1):
        # RRF results are (Document, score) tuples; raw results are Document objects
        if isinstance(item, tuple):
            doc, score = item
            print(f"\n[Rank {rank}] ➔ RRF Score: {score:.6f}")
        else:
            doc = item
            print(f"\n[Rank {rank}]")

        # Display provenance metadata (file source, page, chunk number)
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", "unknown")
        print(f"  Source : {source} | Page: {page} | Chunk: {chunk_idx}")
        
        # Display a clean text preview
        preview = doc.page_content.strip().replace("\n", " ")
        print(f"  Snippet: {preview[:300]}...")


# ------------------------------------------------------------------------------
# 9. Main Execution & Interactive Testing
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "#" * 75)
    print("   DAY 37: HYBRID SEARCH SYSTEM (BM25 + CHROMA + RRF)")
    print("#" * 75)
    print("💡 Try these test queries to see hybrid search in action:")
    print("  1. Exact keyword match : '429' or 'Idempotency-Key'")
    print("     -> BM25 excels at pinning down the exact code or header.")
    print("  2. Conceptual paraphrase : 'prevent duplicate orders when clicking submit twice'")
    print("     -> Semantic search excels by understanding the concept of idempotency.")
    print("  3. Composite query     : 'Idempotency-Key header safe retry mechanism'")
    print("     -> Hybrid search fuses both to give the most accurate chunk at Rank #1.\n")

    # Prompt user for search query
    query = input("Enter your search query: ").strip()

    if not query:
        raise ValueError("Query cannot be empty")

    # Execute Hybrid Search
    bm25_res, semantic_res, hybrid_res = hybrid_search(query=query, top_k=5)

    # Display side-by-side comparison across all 3 search approaches
    display_results("BM25 KEYWORD SEARCH RESULTS", bm25_res)
    display_results("SEMANTIC VECTOR SEARCH RESULTS (CHROMA)", semantic_res)
    display_results("HYBRID FUSION RESULTS (RECIPROCAL RANK FUSION)", hybrid_res)