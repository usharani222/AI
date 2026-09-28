import hashlib
import logging
import time
from typing import Optional

from langchain_core.documents import Document


# ---------------------------------------------------------
# 1. Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("production_rag")


# ---------------------------------------------------------
# 2. Demo identities and permissions
# ---------------------------------------------------------

# In production, derive these values from verified authentication
# and authorization middleware—not from user-provided request fields.

USERS = {
    "user_a": {
        "tenant_id": "team_backend",
        "allowed": True
    },
    "user_b": {
        "tenant_id": "team_frontend",
        "allowed": True
    },
    "user_unauthorized": {
        "tenant_id": "team_backend",
        "allowed": False
    }
}


def get_authorized_scope(user_id: str) -> Optional[str]:
    user = USERS.get(user_id)

    if not user or not user["allowed"]:
        return None

    return user["tenant_id"]


# ---------------------------------------------------------
# 3. Response cache
# ---------------------------------------------------------

response_cache = {}


def make_cache_key(user_id: str, tenant_id: str, query: str) -> str:
    """
    Cache key is scoped by user, tenant, and normalized query.
    """
    normalized_query = " ".join(query.lower().split())

    raw_key = f"{user_id}|{tenant_id}|{normalized_query}"

    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


# ---------------------------------------------------------
# 4. Scoped retrieval
# ---------------------------------------------------------

def retrieve_scoped(query: str, tenant_id: str, k: int = 5):
    """
    Every retrieval is filtered to the authorized tenant.

    Assumes each indexed document has metadata:
        tenant_id
    """
    from langchain_chroma import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings=HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store=Chroma(
            collection_name="Prod-rag",
            embedding_function=embeddings,
    )
    return vector_store.similarity_search(
        query=query,
        k=k,
        filter={"tenant_id": tenant_id}
    )


# ---------------------------------------------------------
# 5. Context builder
# ---------------------------------------------------------

def build_context(documents: list[Document]) -> str:
    context_parts = []

    for index, doc in enumerate(documents, start=1):
        context_parts.append(
            f"[S{index}]\n{doc.page_content}"
        )

    return "\n\n".join(context_parts)


# ---------------------------------------------------------
# 6. Production RAG request
# ---------------------------------------------------------

def answer_query(user_id: str, query: str) -> dict:
    start_time = time.perf_counter()

    if not query.strip():
        return {
            "status": "error",
            "message": "Query cannot be empty."
        }

    # Authorization must happen before cache lookup.
    tenant_id = get_authorized_scope(user_id)

    if tenant_id is None:
        logger.warning("Access denied for user=%s", user_id)

        return {
            "status": "no_access",
            "message": "You are not authorized to access this knowledge base."
        }

    cache_key = make_cache_key(user_id, tenant_id, query)

    # Check cache only after authorization.
    if cache_key in response_cache:
        elapsed = time.perf_counter() - start_time

        logger.info(
            "CACHE HIT user=%s elapsed=%.4fs",
            user_id,
            elapsed
        )

        return {
            "status": "success",
            "answer": response_cache[cache_key],
            "cached": True
        }

    logger.info("CACHE MISS user=%s", user_id)

    # Retrieval may fail if the vector store is unavailable.
    try:
        documents = retrieve_scoped(
            query=query,
            tenant_id=tenant_id,
            k=5
        )

    except Exception:
        logger.exception("Vector store retrieval failed")

        return {
            "status": "error",
            "message": "Knowledge search is temporarily unavailable. Please try again."
        }

    if not documents:
        return {
            "status": "success",
            "answer": "I couldn't find relevant information in your authorized knowledge base.",
            "cached": False
        }

    context = build_context(documents)

    # Count/log actual LLM invocations to verify cache behavior.
    logger.info("LLM CALL user=%s", user_id)

    try:
        answer = chain.invoke({
            "context": context,
            "query": query
        })

    except Exception:
        logger.exception("LLM generation failed")

        return {
            "status": "error",
            "message": "Answer generation is temporarily unavailable. Please try again."
        }

    response_cache[cache_key] = answer

    elapsed = time.perf_counter() - start_time

    logger.info(
        "REQUEST COMPLETE user=%s elapsed=%.4fs",
        user_id,
        elapsed
    )

    return {
        "status": "success",
        "answer": answer,
        "cached": False
    }


# ---------------------------------------------------------
# 7. Test cache, access control, and outage behavior
# ---------------------------------------------------------

if __name__ == "__main__":

    test_query = "What does HTTP 429 mean?"

    print("\n--- First request: expected cache miss and LLM call ---")
    result_1 = answer_query("user_a", test_query)
    print(result_1)

    print("\n--- Second request: expected cache hit, no LLM call ---")
    result_2 = answer_query("user_a", test_query)
    print(result_2)

    print("\n--- Unauthorized user: expected no_access ---")
    result_3 = answer_query("user_unauthorized", test_query)
    print(result_3)

    print("\n--- Different tenant: separate cache scope ---")
    result_4 = answer_query("user_b", test_query)
    print(result_4)