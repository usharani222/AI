import os
import sys
import json
import re
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# LangChain components
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ------------------------------------------------------------------------------
# 1. Configuration & Constants
# ------------------------------------------------------------------------------
EVAL_FILE = Path("eval_set.json")
OUTPUT_FILE = Path("eval_results.json")
TOP_K = 5

# ------------------------------------------------------------------------------
# 2. Setup Vector Store & LLM Chain (Pipeline Definition)
# ------------------------------------------------------------------------------
print("Initializing Embedding Model and Vector Store...")
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to the persisted Chroma vector store created in earlier exercises
vector_store = Chroma(
    collection_name="RAG",
    persist_directory="./vector_store",
    embedding_function=embeddings_model,
)

# Check if the vector store contains documents
chunk_count = vector_store._collection.count()
print(f"Connected to Chroma collection 'RAG' with {chunk_count} indexed chunks.")

if chunk_count == 0:
    print("Warning: Collection 'RAG' is empty. Ingesting PDFs from ./data/pdf...")
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = DirectoryLoader("./data/pdf", glob="**/*.pdf", loader_cls=PyPDFLoader)
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(raw_docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
    vector_store.add_documents(chunks)
    print(f"Ingested {len(chunks)} chunks into vector store.")

# Configure LLM with Citation Instructions
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    print("Warning: OPENROUTER_API_KEY not found in environment. Please set it in .env")

llm = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key or "placeholder_key",
    temperature=0,
)

# The prompt forces the LLM to ground its response in the provided sources
# and cite every statement with [S1], [S2], etc.
citation_prompt = ChatPromptTemplate.from_template("""
You are a factual technical assistant answering questions using ONLY the provided sources.

Rules:
1. Answer the question directly using facts strictly from the provided context.
2. Every claim or sentence MUST cite its supporting source using its exact bracketed label, such as [S1] or [S2].
3. Do NOT cite sources that do not contain the fact.
4. If the context does not contain enough information, state:
   "I couldn't find sufficient information in the provided sources."

Context:
{context}

Question:
{query}

Answer (with citations):
""")

chain = citation_prompt | llm | StrOutputParser()


# ------------------------------------------------------------------------------
# 3. Evaluation Functions
# ------------------------------------------------------------------------------

def load_eval_set():
    """
    Loads the evaluation questions, reference answers, and expected source files.
    """
    if not EVAL_FILE.exists():
        raise FileNotFoundError(
            f"Evaluation file '{EVAL_FILE}' not found. Please create it or provide test questions."
        )

    with EVAL_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def retrieve_with_sources(query: str, k: int = TOP_K):
    """
    Retrieves the top-k chunks from Chroma and formats them with labeled
    citations: [S1], [S2], ..., [Sk].
    
    Returns:
        context (str): The concatenated string of labeled chunks.
        source_map (dict): Maps each label ('S1', 'S2', ...) to metadata:
                           {'source': filename, 'page': page_num, 'chunk_index': idx}
    """
    results = vector_store.similarity_search(query, k=k)

    context_parts = []
    source_map = {}

    for index, doc in enumerate(results, start=1):
        label = f"S{index}"

        # Extract only the file name (e.g. '03_http_status_codes.pdf')
        source = Path(
            doc.metadata.get("source", "unknown")
        ).name

        page = doc.metadata.get("page", "unknown")
        chunk_index = doc.metadata.get("chunk_index", "unknown")

        context_parts.append(
            f"[{label}]\n{doc.page_content}"
        )

        source_map[label] = {
            "source": source,
            "page": page,
            "chunk_index": chunk_index
        }

    context = "\n\n".join(context_parts)
    return context, source_map


def extract_cited_labels(answer: str) -> set:
    """
    Extracts all bracketed citation labels from the generated answer.
    Example: 'HTTP 429 indicates rate limiting [S1].' -> {'S1'}
    """
    return set(re.findall(r"\[(S\d+)\]", answer))


def run_pipeline(query: str, context: str) -> str:
    """
    Executes the RAG chain with context and user query.
    Expected chain input keys: 'context', 'query'
    """
    return chain.invoke({
        "context": context,
        "query": query
    })


def evaluate():
    """
    Main evaluation loop:
    1. Iterates through each question in eval_set.json.
    2. Runs retrieval to build labeled context [S1], [S2], ...
    3. Prompts LLM to generate an answer with citations.
    4. Automatically verifies whether cited source documents match expected sources.
    5. Solicits human feedback for answer accuracy and citation correctness.
    6. Produces aggregate precision/pass metrics and saves detailed logs to eval_results.json.
    """
    eval_questions = load_eval_set()
    results = []

    print(f"\nStarting evaluation of {len(eval_questions)} questions from '{EVAL_FILE}'...\n")

    for item in eval_questions:
        question_id = item["id"]
        question = item["question"]

        print(f"\n{'=' * 70}")
        print(f"📌 {question_id}: {question}")
        print("=" * 70)

        # 1. Retrieval
        context, source_map = retrieve_with_sources(question)

        # 2. Generation
        answer = run_pipeline(question, context)

        # 3. Citation Extraction
        cited_labels = extract_cited_labels(answer)

        # Map cited labels ([S1], [S2]) to actual file names ('03_http_status_codes.pdf')
        cited_sources = {
            source_map[label]["source"]
            for label in cited_labels
            if label in source_map
        }

        expected_sources = set(item.get("expected_sources", []))

        # 4. Automated Check: Did the LLM cite the required documents?
        citation_source_match = (
            expected_sources.issubset(cited_sources)
            if expected_sources
            else len(cited_sources) == 0
        )

        print("\n🤖 Generated Answer:")
        print(answer)

        print("\n🎯 Expected Answer:")
        print(item.get("expected_answer", "(None specified)"))

        print("\n📚 Expected Sources:")
        print(sorted(expected_sources))

        print("\n🏷️ Cited Sources in Generated Answer:")
        print(sorted(cited_sources))

        print("\n🔍 Retrieved Source Map (Top Chunks):")
        print(json.dumps(source_map, indent=2))

        print("\n⚡ Initial Citation Source Match (Automated Subset Check):", citation_source_match)

        # 5. Human Evaluation: Judge correctness against expected ground truth
        try:
            answer_correct = input(
                "\nIs the generated answer factually correct? (y/n): "
            ).strip().lower() == "y"

            citation_correct = input(
                "Are the citations correct and properly supporting the claim? (y/n): "
            ).strip().lower() == "y"
        except (EOFError, KeyboardInterrupt):
            # Fallback if running non-interactively
            print("\n(Non-interactive mode detected: auto-assigning based on automated source match)")
            answer_correct = citation_source_match
            citation_correct = citation_source_match

        results.append({
            "id": question_id,
            "question": question,
            "expected_answer": item.get("expected_answer"),
            "generated_answer": answer,
            "expected_sources": sorted(expected_sources),
            "cited_sources": sorted(cited_sources),
            "answer_correct": answer_correct,
            "citation_correct": citation_correct,
            "initial_citation_source_match": citation_source_match
        })

    # 6. Aggregate Metrics Calculation
    total = len(results)
    answer_passes = sum(result["answer_correct"] for result in results)
    citation_passes = sum(result["citation_correct"] for result in results)

    answer_rate = (answer_passes / total) * 100 if total else 0
    citation_rate = (citation_passes / total) * 100 if total else 0

    summary = {
        "total_questions": total,
        "answer_correct": answer_passes,
        "answer_pass_rate": round(answer_rate, 2),
        "citation_correct": citation_passes,
        "citation_correct_rate": round(citation_rate, 2),
        "results": results
    }

    # 7. Write Results to Disk
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print("📊 RAG EVALUATION BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total Evaluated Questions : {total}")
    print(f"Answer Correctness Rate   : {answer_passes}/{total} ({answer_rate:.1f}%)")
    print(f"Citation Correctness Rate : {citation_passes}/{total} ({citation_rate:.1f}%)")
    print(f"Detailed results saved to : {OUTPUT_FILE.resolve()}")
    print("=" * 70)


# ------------------------------------------------------------------------------
# 4. Entry Point
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    evaluate()
