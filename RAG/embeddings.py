
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# 1. Sample document
document = """
EventFlow is a multi-tenant event booking platform.

Users can browse events and reserve available seats.

Redis temporarily holds seats during checkout.
Each reservation expires after five minutes.

Once payment succeeds, the booking is confirmed
and stored in PostgreSQL.

If payment fails, the temporary seat reservation
must be released so another user can book it.
"""


# 2. Chunk the document
splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30,
    separators=["\n\n", "\n", " ", ""],
)

chunks = splitter.split_text(document)

print("Total chunks:", len(chunks))


# 3. Initialize embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# 4. Attach metadata to every chunk
texts = chunks

metadatas = [
    {
        "source": "eventflow_document.txt",
        "chunk_index": i,
        "team_id": "engineering",
    }
    for i in range(len(chunks))
]


# 5. Store vectors and metadata in Chroma
vector_store = Chroma.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas,
    collection_name="eventflow_day31",
    persist_directory="./chroma_db",
)


print("Chunks embedded and stored successfully!")


# 6. Inspect stored vectors
sample_vector = embeddings.embed_query(chunks[0])

print("Embedding dimensions:", len(sample_vector))
print("First 5 values:", sample_vector[:5])


# 7. Perform similarity search
query = "How long does a seat reservation last?"
# "Where are confirmed bookings stored?"
# "What happens when payment fails?"
# "What is the purpose of Redis in EventFlow?"
# "How does a user reserve an event seat?"
results = vector_store.similarity_search_with_score(
    query,
    k=3,
)

print("\nRetrieved results:")

for doc, score in results:
    print("\nContent:", doc.page_content)
    print("Metadata:", doc.metadata)
    print("Distance:", score)