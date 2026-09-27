import os
import sys
import shutil
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

# ==============================================================================
# DAY 36 — AI: RAG Metadata Filtering & Access Control
# ==============================================================================
# Security Concept:
# - Filtering at the RETRIEVAL layer is a hard boundary: unauthorized documents
#   are NEVER fetched from the vector DB, NEVER sent to the LLM, and CANNOT
#   leak through prompt injection or model hallucination.
# - Prompt-level instructions (e.g. "Do not reveal financial secrets") are soft
#   and can easily be bypassed by clever user inputs.
# ==============================================================================

# ------------------------------------------------------------------------------
# 1. Prepare Multi-Tenant Raw Documents
# ------------------------------------------------------------------------------
raw_documents = [
    # Engineering documents
    Document(
        page_content=(
            "Engineering Roadmap Q4: We are migrating all backend microservices to "
            "Kubernetes on AWS EKS. Planned server upgrade budget is $25,000 for cloud "
            "infrastructure. All API endpoints will mandate OAuth2 / mTLS authentication."
        ),
        metadata={
            "source": "eng_roadmap_q4.md",
            "team": "engineering",
            "doc_type": "technical_plan",
            "access_level": "team_only",
        },
    ),
    Document(
        page_content=(
            "DevOps Deployment Guide: Production database maintenance happens every Sunday "
            "at 02:00 UTC. The Redis cluster handles temporary seat reservation caching "
            "with a TTL of 300 seconds."
        ),
        metadata={
            "source": "devops_playbook.md",
            "team": "engineering",
            "doc_type": "guide",
            "access_level": "team_only",
        },
    ),
    # Finance documents
    Document(
        page_content=(
            "Finance Department Q4 Budget Report: Total allocated company expansion budget "
            "is $500,000. Employee salary revisions and bonus disbursements are scheduled "
            "for December 15. The quarterly profit margin increased by 14%."
        ),
        metadata={
            "source": "q4_financial_report.md",
            "team": "finance",
            "doc_type": "financial_report",
            "access_level": "confidential",
        },
    ),
    Document(
        page_content=(
            "Executive Compensation & Payroll Policy: Annual executive bonuses are pegged "
            "to EBITDA targets. All reimbursement claims exceeding $1,000 require CFO sign-off."
        ),
        metadata={
            "source": "payroll_policy.md",
            "team": "finance",
            "doc_type": "policy",
            "access_level": "confidential",
        },
    ),
    # Shared / Company-wide documents
    Document(
        page_content=(
            "Company-wide All Hands Overview: The company is growing rapidly across both "
            "Engineering and Finance teams. Upcoming holiday break starts on December 24. "
            "All employees must complete mandatory cybersecurity training by November 30."
        ),
        metadata={
            "source": "company_handbook.md",
            "team": "all",
            "doc_type": "general",
            "access_level": "public",
        },
    ),
]

# ------------------------------------------------------------------------------
# 2. Chunk Documents & Preserve/Enrich Metadata
# ------------------------------------------------------------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=40,
    length_function=len,
)

chunks = splitter.split_documents(raw_documents)

# Assign a deterministic chunk index to metadata
for idx, chunk in enumerate(chunks):
    chunk.metadata["chunk_id"] = idx

print(f"Created {len(chunks)} chunks across {len(raw_documents)} raw documents.")

# ------------------------------------------------------------------------------
# 3. Setup Embedding Model & Chroma Vector Store
# ------------------------------------------------------------------------------
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Use a dedicated collection and directory for metadata filtering practice
DB_DIR = "./vector_store_metadata"

vector_store = Chroma(
    collection_name="team_scoped_rag",
    embedding_function=embeddings_model,
    persist_directory=DB_DIR,
)

# Avoid duplicate insertions by providing deterministic IDs and checking count
if vector_store._collection.count() == 0:
    deterministic_ids = [
        f"{c.metadata['team']}_{c.metadata['source']}_{i}"
        for i, c in enumerate(chunks)
    ]
    vector_store.add_documents(chunks, ids=deterministic_ids)
    print(f"Ingested {len(chunks)} chunks into vector store.")
else:
    print(f"Vector store already initialized with {vector_store._collection.count()} chunks.")

# ------------------------------------------------------------------------------
# 4. Metadata-Filtered Retrieval
# ------------------------------------------------------------------------------
def retrieve_for_user(query: str, user_team: str, k: int = 3):
    """
    Retrieves chunks with strict metadata filtering applied AT THE RETRIEVAL LAYER.
    A user in 'engineering' can only see chunks where team is 'engineering' or 'all'.
    """
    # Chroma filter syntax:
    # Filter by exact match or $in operator for shared documents
    query_filter = {
        "$or": [
            {"team": {"$eq": user_team}},
            {"team": {"$eq": "all"}},
        ]
    }

    results = vector_store.similarity_search_with_score(
        query=query,
        k=k,
        filter=query_filter,
    )
    return results

# ------------------------------------------------------------------------------
# 5. LLM Chain Setup
# ------------------------------------------------------------------------------
prompt_template = ChatPromptTemplate.from_template("""
You are an AI assistant answering strictly from the authorized context provided.
Do not use outside knowledge or speculate.

If the context does not contain enough information, state:
"I do not have access to sufficient information in your authorized documents."

Context:
{context}

Question:
{question}

Answer:
""")

llm = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
)

rag_chain = prompt_template | llm | StrOutputParser()

def answer_for_user(user_name: str, user_team: str, question: str):
    """Executes end-to-end RAG scoped to the user's team permissions."""
    print(f"\n{'='*70}")
    print(f"👤 User: {user_name} | Scope: Team [{user_team.upper()}]")
    print(f"❓ Question: '{question}'")
    print(f"{'='*70}")

    retrieved_items = retrieve_for_user(question, user_team=user_team, k=3)

    if not retrieved_items:
        print("No documents matched the query within this user's access scope.")
        return

    context_parts = []
    print("\n🔍 Retrieved Chunks (Security Boundary Check):")
    for i, (doc, score) in enumerate(retrieved_items, start=1):
        team_tag = doc.metadata.get("team")
        source = doc.metadata.get("source")
        print(f"  [{i}] Source: {source} | Team Tag: {team_tag} | Distance: {score:.4f}")
        context_parts.append(f"[Source: {source} (Team: {team_tag})]\n{doc.page_content}")

    context_text = "\n\n".join(context_parts)

    answer = rag_chain.invoke({
        "context": context_text,
        "question": question,
    })

    print(f"\n💬 Answer for {user_name}:")
    print(answer)

# ------------------------------------------------------------------------------
# 6. Verification: Same Question, Different Access Scopes
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Question inquiring about plans and budgets
    test_query = "What are the upcoming budget and upgrade plans?"

    # 1. User Alice (Engineering)
    answer_for_user(
        user_name="Alice (Lead Software Engineer)",
        user_team="engineering",
        question=test_query,
    )

    # 2. User Bob (Finance Manager)
    answer_for_user(
        user_name="Bob (Finance Analyst)",
        user_team="finance",
        question=test_query,
    )
