
from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer



#Load the document


file_path = Path(__file__).parent / "document.txt"

text = file_path.read_text(
    encoding="utf-8"
)



# Split the document into chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

print("Number of chunks:", len(chunks))



#Convert chunks into Document objects

documents = []

for i, chunk in enumerate(chunks, start=1):

    document = Document(
        page_content=chunk,
        metadata={
            "source": "document.txt",
            "chunk_id": i
        }
    )

    documents.append(document)



#Create embeddings


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = embedding_model.encode(
    [document.page_content for document in documents]
)



# Create FAISS index


dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(
    np.array(embeddings).astype("float32")
)

print("Vectors stored in FAISS:", index.ntotal)



# Ask a question
-

question = input(
    "\nAsk a question about the document: "
).strip()

if not question:
    print("Please enter a question.")
    exit()



# Create question embedding


question_embedding = embedding_model.encode(
    [question]
)



# Search FAISS


distances, indices = index.search(
    np.array(question_embedding).astype("float32"),
    k=2
)



# Retrieve the Documents

print("\nRetrieved documents:")

for distance, index_number in zip(
    distances[0],
    indices[0]
):

    document = documents[index_number]

    print("\n--------------------")

    print("Distance:")
    print(distance)

    print("\nPage content:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)

