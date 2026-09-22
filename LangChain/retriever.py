from langchain_core.documents import Document


class ToyRetriever:
    def __init__(self, documents):
        self.documents = documents

    def invoke(self, query: str) -> list[Document]:
        query = query.lower()

        results = []

        for document in self.documents:
            text = document.page_content.lower()

            # Simple keyword matching
            query_words = query.split()

            if any(word in text for word in query_words):
                results.append(document)

        return results


# --------------------------------------------------
# Our small document collection
# --------------------------------------------------

documents = [
    Document(
        page_content="Redis is an in-memory data store commonly used for caching.",
        metadata={"topic": "redis"}
    ),

    Document(
        page_content="Redis can also be used for sessions and message processing.",
        metadata={"topic": "redis"}
    ),

    Document(
        page_content="PostgreSQL is a relational database management system.",
        metadata={"topic": "postgresql"}
    ),

    Document(
        page_content="PostgreSQL supports SQL queries, transactions, and indexes.",
        metadata={"topic": "postgresql"}
    ),

    Document(
        page_content="Docker packages applications and their dependencies into containers.",
        metadata={"topic": "docker"}
    ),

    Document(
        page_content="Docker containers make applications easier to deploy consistently.",
        metadata={"topic": "docker"}
    ),

    Document(
        page_content="FastAPI is a Python framework for building REST APIs.",
        metadata={"topic": "fastapi"}
    ),

    Document(
        page_content="FastAPI uses Python type hints for request validation and API schemas.",
        metadata={"topic": "fastapi"}
    ),
]


# --------------------------------------------------
# Create retriever
# --------------------------------------------------

retriever = ToyRetriever(documents)


# --------------------------------------------------
# Test 1
# --------------------------------------------------

print("\n===== TEST 1 =====")

query = "Redis caching"

results = retriever.invoke(query)

print("Query:", query)

for document in results:
    print(document.page_content)
    print("Metadata:", document.metadata)


# --------------------------------------------------
# Test 2
# --------------------------------------------------

print("\n===== TEST 2 =====")

query = "PostgreSQL database"

results = retriever.invoke(query)

print("Query:", query)

for document in results:
    print(document.page_content)
    print("Metadata:", document.metadata)


# --------------------------------------------------
# Test 3
# --------------------------------------------------

print("\n===== TEST 3 =====")

query = "Docker containers"

results = retriever.invoke(query)

print("Query:", query)

for document in results:
    print(document.page_content)
    print("Metadata:", document.metadata)