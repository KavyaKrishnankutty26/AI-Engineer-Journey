
import os
from pathlib import Path

import faiss
import numpy as np

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from sentence_transformers import SentenceTransformer



#  Load environment variables


load_dotenv()



# Find the document

file_path = Path(__file__).parent / "Document.txt"



# Load the document


loader = TextLoader(
    str(file_path),
    encoding="utf-8"
)

documents = loader.load()

print("Documents loaded:", len(documents))



# Create the text splitter


splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=20
)



# Split Documents into chunks


chunks = splitter.split_documents(
    documents
)

print("Chunks created:", len(chunks))



# Create the embedding model


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)



# Get text from every chunk


chunk_texts = [
    chunk.page_content
    for chunk in chunks
]



# Create embeddings

embeddings = embedding_model.encode(
    chunk_texts
)



# Create FAISS index


dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)



# Store embeddings


index.add(
    np.array(embeddings).astype("float32")
)

print("Vectors stored in FAISS:", index.ntotal)



# Get user's question


question = input(
    "\nAsk a question about the document: "
).strip()



# Validate question


if not question:

    print("Please enter a question.")

    exit()



# Convert question into an embedding


question_embedding = embedding_model.encode(
    [question]
)



#  Search FAISS


distances, indices = index.search(
    np.array(question_embedding).astype("float32"),
    k=2
)




#  Collect retrieved Documents


retrieved_documents = []

for distance, index_number in zip(
    distances[0],
    indices[0]
):

    document = chunks[index_number]

    retrieved_documents.append(document)



# Create context for Gemini


context = "\n\n".join(
    document.page_content
    for document in retrieved_documents
)


print("\nRetrieved context:")
print(context)



# Create Gemini model


model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite"
)



# Create RAG prompt

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a helpful AI assistant.

Answer the user's question using ONLY
the provided context.

If the answer cannot be found in the context,
say:

"The information is not available in the document."

Do not invent information.
"""
        ),
        (
            "human",
            """
Context:
{context}

Question:
{question}
"""
        )
    ]
)



# Create the chain

chain = prompt | model



# Ask Gemini


response = chain.invoke(
    {
        "context": context,
        "question": question
    }
)



# Display the final answer

print("\nAI Answer:")
print(response.content)

