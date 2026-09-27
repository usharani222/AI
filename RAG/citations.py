import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# --------------------------------------------------
# 1. Load the same embedding model and vector store
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name="my_document",
    persist_directory="./vector_store",
    embedding_function=embeddings,
)

# --------------------------------------------------
# 2. Configure the LLM
# --------------------------------------------------

llm = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0,
)

prompt = ChatPromptTemplate.from_template("""
You answer questions using ONLY the supplied context.

Do not use outside knowledge.
If the context does not support an answer, say:
"I couldn't find sufficient information in the provided sources."

Context:
{context}

Question:
{question}

Answer:
""")

answer_chain = prompt | llm | StrOutputParser()

# --------------------------------------------------
# 3. Configure retrieval threshold
# --------------------------------------------------

TOP_K = 5

# Example starting point only.
# Calibrate this using your own relevant and irrelevant queries.
MAX_DISTANCE = 1.0


def answer_question(question: str):
    results = vector_store.similarity_search_with_score(
        question,
        k=TOP_K,
    )

    # Keep only results that pass the distance threshold
    accepted_results = [
        (doc, distance)
        for doc, distance in results
        if distance <= MAX_DISTANCE
    ]

    # No retrieved chunks passed the threshold
    if not accepted_results:
        return {
            "answer": "I couldn't find sufficient information in the provided sources.",
            "sources": [],
            "status": "not_found",
        }

    # Format context and collect citations from real metadata
    context_parts = []
    sources = []

    for i, (doc, distance) in enumerate(accepted_results, start=1):
        source = doc.metadata.get("source", "unknown source")
        page = doc.metadata.get("page")
        chunk_index = doc.metadata.get("chunk_index")

        citation = {
            "source": source,
            "page": page,
            "chunk_index": chunk_index,
        }

        label = f"S{i}"

        context_parts.append(
            f"[{label}]\n"
            f"Source: {source}\n"
            f"Page: {page}\n"
            f"Chunk: {chunk_index}\n"
            f"Content: {doc.page_content}"
        )

        sources.append(citation)

    context = "\n\n".join(context_parts)

    answer = answer_chain.invoke({
        "context": context,
        "question": question,
    })

    return {
        "answer": answer,
        "sources": sources,
        "status": "answered",
    }


# --------------------------------------------------
# 4. Ask a question
# --------------------------------------------------

if __name__ == "__main__":
    question = input("Enter your question: ")

    result = answer_question(question)

    print("\nAnswer:")
    print(result["answer"])

    print("\nSources:")
    if result["sources"]:
        for source in result["sources"]:
            print(source)
    else:
        print("No source passed the relevance threshold.")