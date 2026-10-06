
from pathlib import Path

import faiss
import numpy as np

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# 1. Find the document
# --------------------------------------------------

file_path = Path(__file__).parent / "Document.txt"


# --------------------------------------------------
# 2. Load the document
# --------------------------------------------------

loader = TextLoader(
    str(file_path),
    encoding="utf-8"
)

documents = loader.load()

print("Documents loaded:", len(documents))


# --------------------------------------------------
# 3. Create the text splitter
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)


# --------------------------------------------------
# 4. Split Documents into chunks
# --------------------------------------------------

chunks = splitter.split_documents(
    documents
)

print("Chunks created:", len(chunks))


# --------------------------------------------------
# 5. Create the embedding model
# --------------------------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 6. Get text from every chunk
# --------------------------------------------------

chunk_texts = [
    chunk.page_content
    for chunk in chunks
]


# --------------------------------------------------
# 7. Create embeddings for the chunks
# --------------------------------------------------

embeddings = embedding_model.encode(
    chunk_texts
)


# --------------------------------------------------
# 8. Create FAISS index
# --------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)


# --------------------------------------------------
# 9. Store embeddings in FAISS
# --------------------------------------------------

index.add(
    np.array(embeddings).astype("float32")
)

print("Vectors stored in FAISS:", index.ntotal)


# --------------------------------------------------
# 10. Get user's question
# --------------------------------------------------

question = input(
    "\nAsk a question about the document: "
).strip()


# --------------------------------------------------
# 11. Validate question
# --------------------------------------------------

if not question:

    print("Please enter a question.")

    exit()


# --------------------------------------------------
# 12. Convert question into an embedding
# --------------------------------------------------

question_embedding = embedding_model.encode(
    [question]
)


# --------------------------------------------------
# 13. Search FAISS
# --------------------------------------------------

distances, indices = index.search(
    np.array(question_embedding).astype("float32"),
    k=2
)


# --------------------------------------------------
# 14. Display retrieved Documents
# --------------------------------------------------

print("\nRetrieved documents:")


for distance, index_number in zip(
    distances[0],
    indices[0]
):

    document = chunks[index_number]

    print("\n--------------------")

    print("Distance:")
    print(distance)

    print("\nPage content:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)

