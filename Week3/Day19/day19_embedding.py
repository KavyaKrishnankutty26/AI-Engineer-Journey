
from pathlib import Path

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
# 4. Split Documents into smaller Documents
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
# 6. Get the text from every chunk
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


# --------------------------------------------------
# 8. Display embedding information
# --------------------------------------------------

print("\nEmbedding shape:")
print(embeddings.shape)

print("\nNumber of embeddings:")
print(len(embeddings))

print("\nEmbedding dimensions:")
print(embeddings.shape[1])

