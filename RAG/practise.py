from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

# 1. Load PDFs
loader = DirectoryLoader(
    path="./data/pdf",
    glob="**/*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True
)

documents = loader.load()
print(f"Loaded {len(documents)} pages")

# 2. Split documents
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""],
    length_function=len
)

chunks = splitter.split_documents(documents)

# 3. Add chunk metadata
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_index"] = i

print(f"Created {len(chunks)} chunks")

# 4. Generate embeddings
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 5. Create vector store and add chunks
vector_store = Chroma(
    collection_name="RAG",
    embedding_function=embeddings_model,
    persist_directory="./vector_store"
)

vector_store.add_documents(chunks)

# 6. Retrieve top-k chunks
def retriever(query):
    return vector_store.similarity_search_with_score(
        query=query,
        k=3
    )

# 7. Build context and collect source metadata
def build_context(results):
    context_parts = []
    sources = []

    for i, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "unknown")
        chunk_index = doc.metadata.get("chunk_index", "unknown")

        context_parts.append(
            f"[S{i}]\n{doc.page_content}"
        )

        sources.append({
            "label": f"S{i}",
            "source": source,
            "page": page,
            "chunk_index": chunk_index,
            "distance": score
        })

    return "\n\n".join(context_parts), sources

# 8. Prompt template
prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template="""
You are a helpful assistant.

Answer using only the supplied context.
If the context does not contain enough information, say:
"I don't have enough information in the provided documents to answer that."
Cite supporting source labels such as [S1] or [S2].
Do not invent source labels.

Context:
{context}

Question:
{query}

Answer:
"""
)

# 9. Load model
load_dotenv()

model = ChatOpenAI(
    model="openrouter/free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# 10. Create chain
chain = prompt_template | model | StrOutputParser()

# 11. Ask question
question = input("Enter question: ")

retrieved_results = retriever(question)
context, sources = build_context(retrieved_results)

result = chain.invoke({
    "context": context,
    "query": question
})

print("\nAnswer:")
print(result)

print("\nRetrieved sources:")
for source in sources:
    print(source)