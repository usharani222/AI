import os
import re

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_openai import ChatOpenAI
from langchain.tools import tool


# Load variables from .env file
load_dotenv()


# Words that usually do not provide useful information for
# our simple keyword-based retrieval.
#
# This is only for our toy retriever.
# Production retrievers use embeddings and vector search.
STOP_WORDS = {
    "why", "what", "is", "are", "the", "a", "an", "how", "does", "do",
    "used", "for", "in", "to", "of", "and", "who", "where", "when", "which", "can", "will"
}


# ---------------------------------------------------------
# 1. TOY RETRIEVER
# ---------------------------------------------------------

class ToyRetriever:

    def __init__(self, documents):
        # Store the knowledge-base documents
        self.documents = documents

    def invoke(self, query: str) -> list[Document]:
        # Convert query to lowercase and extract words without punctuation
        # (e.g. "PostgreSQL?" -> "postgresql")
        words = re.findall(r"\b\w+\b", query.lower())

        # Remove common words that are not useful for keyword matching
        query_words = [
            word
            for word in words
            if word not in STOP_WORDS
        ]

        results = []

        # Check every document in our knowledge base
        for document in self.documents:
            text = document.page_content.lower()

            # If ANY query keyword appears in the document,
            # consider that document relevant.
            if any(word in text for word in query_words):
                results.append(document)

        return results


# ---------------------------------------------------------
# 2. EVENTFLOW KNOWLEDGE BASE
# ---------------------------------------------------------

documents = [
    Document(
        page_content="EventFlow is a multi-tenant event booking platform.",
        metadata={"topic": "About"}
    ),
    Document(
        page_content="Events can have reserved seating.",
        metadata={"topic": "Reserved Seating"}
    ),
    Document(
        page_content="Redis is used to temporarily hold seats during booking.",
        metadata={"topic": "Redis"}
    ),
    Document(
        page_content="PostgreSQL stores users, events, and booking information.",
        metadata={"topic": "PostgreSQL"}
    ),
    Document(
        page_content="A seat hold prevents another user from booking the same seat temporarily.",
        metadata={"topic": "Seat Hold"}
    ),
    Document(
        page_content="Payment must be confirmed before a booking becomes confirmed.",
        metadata={"topic": "Payment"}
    ),
    Document(
        page_content="Redis locks are used to prevent concurrent seat reservations.",
        metadata={"topic": "Redis"}
    ),
    Document(
        page_content="The booking service handles booking creation and confirmation.",
        metadata={"topic": "Booking Service"}
    ),
    Document(
        page_content="MongoDB is used for notification-related data.",
        metadata={"topic": "MongoDB"}
    ),
    Document(
        page_content="The platform supports event creators and attendees.",
        metadata={"topic": "Event creators and attendees"}
    )
]


# ---------------------------------------------------------
# 3. CREATE RETRIEVER
# ---------------------------------------------------------

# Create an instance of our toy retriever using EventFlow documents
retriever = ToyRetriever(documents)


# ---------------------------------------------------------
# 4. TOOL
# ---------------------------------------------------------

@tool
def get_service_status(service_name: str) -> str:
    """Check the real-time operational status and health metrics of an EventFlow internal service (e.g., redis, postgresql, mongodb, booking)."""
    service = service_name.lower().strip()
    statuses = {
        "redis": "Operational (0% packet loss, memory 42% used)",
        "postgresql": "Operational (Healthy connections: 28/100)",
        "mongodb": "Operational (Replica set healthy, latency: 3ms)",
        "booking": "Operational (Throughput: 120 req/s, queue depth: 0)"
    }
    return statuses.get(service, f"Service '{service_name}' status is unknown.")


tools = [get_service_status]


# ---------------------------------------------------------
# 5. RETRIEVE CONTEXT
# ---------------------------------------------------------

def get_context(question: str) -> str:
    # Ask the retriever to find documents relevant to the user's question
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return "NO RELEVANT INFORMATION FOUND"

    return "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )


# ---------------------------------------------------------
# 5. PROMPT TEMPLATE
# ---------------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an internal EventFlow knowledge assistant.

Answer the user's question using the provided context.
If asked about live system operational health or service status, call the `get_service_status` tool.

If the context does not contain enough information to answer, say:
"I don't have enough information in the provided knowledge base to answer that."

Do not invent information.

Context:
{context}"""
    ),
    (
        "human",
        "{question}"
    )
])


# ---------------------------------------------------------
# 7. CHAT MODEL & TOOL BINDING
# ---------------------------------------------------------

model = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# Bind tool to the model
model_with_tools = model.bind_tools(tools)


# ---------------------------------------------------------
# 8. RESPONSE HANDLER (OUTPUT PARSER)
# ---------------------------------------------------------

def handle_response(message):
    # If the model chose to invoke a tool, execute it
    if hasattr(message, "tool_calls") and message.tool_calls:
        call = message.tool_calls[0]
        tool_name = call["name"]
        args = call["args"]
        if tool_name == "get_service_status":
            result = get_service_status.invoke(args)
            return f"[Tool Used: {tool_name}] Status: {result}"
        return f"[Tool Used: {tool_name}] Args: {args}"

    # Standard text answer
    return getattr(message, "content", str(message))


# ---------------------------------------------------------
# 9. RUNNABLE PARALLEL & LCEL CHAIN 
# ---------------------------------------------------------

parallel = RunnableParallel(
    context=RunnableLambda(lambda x: get_context(x["question"])),
    question=RunnableLambda(lambda x: x["question"])
)

# LCEL Pipeline:
# parallel -> prompt -> model_with_tools -> handle_response
chain = (
    parallel
    | prompt
    | model_with_tools
    | RunnableLambda(handle_response)
)


# ---------------------------------------------------------
# 10. CHECKPOINT QUESTIONS & VERIFICATION (DAY 21)
# ---------------------------------------------------------
#
# Checkpoint Question:
# "If you swapped your toy retriever for a completely different one tomorrow,
#  would any other part of your assistant need to change?"
#
# Answer:
# No. Because the LCEL chain communicates with the retriever via the standard
# Runnable interface (.invoke(query) -> list[Document]) through `get_context`,
# replacing ToyRetriever with a production vector retriever (e.g. Chroma, FAISS,
# Pinecone) requires ZERO changes to the prompt, model, tools, or LCEL chain.
# ---------------------------------------------------------

if __name__ == "__main__":
    # Test suite:
    # - 3 test questions answered from retrieved context
    # - 1 question with no matching snippet (visibly declines)
    # - 1 question testing tool usage
    test_queries = [
        "What is PostgreSQL used for in EventFlow?",
        "Why is Redis used in the platform?",
        "What does the booking service handle?",
        "Who is the CEO of EventFlow?",
        "What is the live operational status of Redis?"
    ]

    print("\n" + "=" * 70)
    print("EVENTFLOW INTERNAL KNOWLEDGE ASSISTANT (DAY 21 CHECKPOINT)")
    print("=" * 70 + "\n")

    for i, q in enumerate(test_queries, 1):
        print(f"Test {i}: {q}")
        response = chain.invoke({"question": q})
        print(f"Answer: {response}")
        print("-" * 70)