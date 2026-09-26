
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

document = """
Retrieval-Augmented Generation (RAG) combines information retrieval
with large language models.

Chunking is an important preprocessing step in RAG.
It divides large documents into smaller pieces before embedding.

Fixed-size chunking splits text according to a specified character
or token count. It does not inherently understand sentence meaning.

Recursive chunking attempts to preserve structural boundaries.
It tries paragraph separators first, followed by line breaks,
spaces, and finally individual characters.

Poor chunk boundaries can separate important context.
This may reduce retrieval quality even when the embedding model
is capable of producing high-quality representations.

Chunk size and overlap are important configuration parameters.
They influence how much context is preserved in each chunk.
"""

# Strategy 1: Fixed-size splitting
fixed_splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=150,
    chunk_overlap=0,
    length_function=len,
)

fixed_chunks = fixed_splitter.split_text(document)

# Strategy 2: Recursive splitting
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    length_function=len,
    separators=["\n\n", "\n", " ", ""],
)

recursive_chunks = recursive_splitter.split_text(document)


def inspect_chunks(name, chunks):
    print(f"\n{'=' * 60}")
    print(name)
    print(f"Total chunks: {len(chunks)}")
    print("=" * 60)

    for i, chunk in enumerate(chunks, start=1):
        print(f"\nChunk {i}")
        print(f"Length: {len(chunk)} characters")
        print(f"Text: {chunk}")

        # Simple boundary inspection
        print(
            "Ends with sentence punctuation:",
            chunk.rstrip().endswith((".", "!", "?"))
        )


inspect_chunks("FIXED-SIZE CHUNKING", fixed_chunks)
inspect_chunks("RECURSIVE CHUNKING", recursive_chunks)