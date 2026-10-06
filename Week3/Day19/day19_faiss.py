
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
# 7. Create embeddings
# --------------------------------------------------

embeddings = embedding_model.encode(
    chunk_texts
)

print("\nEmbedding shape:")
print(embeddings.shape)


# --------------------------------------------------
# 8. Get embedding dimensions
# --------------------------------------------------

dimension = embeddings.shape[1]

print("\nEmbedding dimensions:")
print(dimension)


# --------------------------------------------------
# 9. Create FAISS index
# --------------------------------------------------

index = faiss.IndexFlatL2(
    dimension
)


# --------------------------------------------------
# 10. Add embeddings to FAISS
# --------------------------------------------------

index.add(
    np.array(embeddings).astype("float32")
)


# --------------------------------------------------
# 11. Display FAISS information
# --------------------------------------------------

print("\nVectors stored in FAISS:")
print(index.ntotal)

