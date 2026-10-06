
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# 1. Load the document
# --------------------------------------------------

file_path = Path(__file__).parent / "document.txt"

text = file_path.read_text(
    encoding="utf-8"
)


# --------------------------------------------------
# 2. Create the text splitter
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)


# --------------------------------------------------
# 3. Split the document into chunks
# --------------------------------------------------

chunks = splitter.split_text(text)

print("Number of chunks:", len(chunks))


# --------------------------------------------------
# 4. Add metadata to each chunk
# --------------------------------------------------

documents = []

for i, chunk in enumerate(chunks, start=1):

    document = {
        "text": chunk,
        "metadata": {
            "source": "document.txt",
            "chunk_id": i
        }
    }

    documents.append(document)


# --------------------------------------------------
# 5. Display chunks and metadata
# --------------------------------------------------

for document in documents:

    print("\n--------------------")

    print("Chunk:")
    print(document["text"])

    print("\nMetadata:")
    print(document["metadata"])

