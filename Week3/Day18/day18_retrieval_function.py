from pathlib import Path

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer



# Load the document

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



# Convert chunks into Document objects


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



# Create the embedding model


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)



# Create embeddings for documents

embeddings = embedding_model.encode(
    [
        document.page_content
        for document in documents
    ]
)



# Create FAISS index

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(
    np.array(embeddings).astype("float32")
)

print("Vectors stored in FAISS:", index.ntotal)



# Create retrieval function

def retrieve_documents(question):

    # Create embedding for the question
    question_embedding = embedding_model.encode(
        [question]
    )

    # Retrieve candidate documents
    distances, indices = index.search(
        np.array(question_embedding).astype("float32"),
        k=4
    )

    # Distance threshold
    threshold = 1.0

    relevant_documents = []

    # Check each candidate
    for distance, index_number in zip(
        distances[0],
        indices[0]
    ):

        print(
            f"\nDistance: {distance:.2f}"
        )

        # Keep only sufficiently relevant documents
        if distance <= threshold:

            document = documents[index_number]

            relevant_documents.append(
                document
            )

    return relevant_documents



# Get user question


question = input(
    "\nAsk a question about the document: "
).strip()

if not question:

    print("Please enter a question.")
    exit()



#  Retrieve relevant documents

retrieved_documents = retrieve_documents(
    question
)



#  Display retrieved documents


print("\nRelevant documents:")

if retrieved_documents:

    for document in retrieved_documents:

        print("\n--------------------")

        print("Page content:")
        print(document.page_content)

        print("\nMetadata:")
        print(document.metadata)

else:

    print("No sufficiently relevant documents found.")

