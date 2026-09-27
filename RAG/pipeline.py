import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

# 1. Load PDF
loader = TextLoader("./data/text_files/sample.txt")
documents = loader.load()

print(f"Document loaded successfully: {len(documents)} pages")

# 2. Chunk documents
splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=200,
    chunk_overlap=20,
    length_function=len,
)

chunks = splitter.split_documents(documents)

print(f"Chunking done: {len(chunks)} chunks")

# 3. Create embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully")

# 4. Create vector store
vector_store = Chroma(
    embedding_function=embeddings,
    collection_name="my_document",
    persist_directory="./vector_store",
)

# Add chunks to the vector store
# Stable IDs help avoid adding duplicate records when rerunning this script.
chunk_ids = [f"sample_pdf_chunk_{i}" for i in range(len(chunks))]
vector_store.add_documents(documents=chunks, ids=chunk_ids)

print("Chunks added to vector store")

# 5. Get user query
query = input("Enter question: ")

# 6. Retrieve relevant chunks
retrieved_results = vector_store.similarity_search_with_score(
    query,
    k=5,
)

# Extract the Document objects from the (Document, score) tuples
retrieved_docs = [doc for doc, score in retrieved_results]

# Format retrieved documents into a context string
context = "\n\n".join(
    f"[Source: {doc.metadata.get('source', 'unknown')}, "
    f"Page: {doc.metadata.get('page', 'unknown')}]\n"
    f"{doc.page_content}"
    for doc in retrieved_docs
)

print("\nRetrieved context:")
print(context)

# 7. Create prompt
prompt = PromptTemplate(
    template="""You are a helpful AI assistant.
Answer the question using ONLY the retrieved context below.

If the answer is not supported by the context, say:
"I don't have enough information in the provided context."

Context:
{context}

Question:
{query}

Answer:""",
    input_variables=["context", "query"],
)

# 8. Create model
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY is missing from your environment")

model = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    temperature=0,
)

# 9. Build and invoke RAG chain
chain = prompt | model | StrOutputParser()

result = chain.invoke({
    "context": context,
    "query": query,
})

print("\nGenerated answer:")
print(result)