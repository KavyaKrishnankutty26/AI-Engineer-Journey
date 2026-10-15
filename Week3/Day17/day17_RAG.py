import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate



# Load environment variables


load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")



# Create the embedding model


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)



# Load the document


file_path = Path(__file__).parent / "document.txt"

text = file_path.read_text(
    encoding="utf-8"
)



# Split the document into simple chunks


chunks = [
    chunk.strip()
    for chunk in text.split("\n\n")
    if chunk.strip()
]


print("Number of chunks:", len(chunks))



# Create embeddings for the chunks


chunk_embeddings = embedding_model.encode(
    chunks
)



# Create FAISS index

dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)



# Store embeddings in FAISS


index.add(
    np.array(chunk_embeddings).astype("float32")
)



# Get user's question


question = input(
    "\nAsk a question about the document: "
).strip()


if not question:
    print("Please enter a question.")
    exit()



# Convert question into an embedding

question_embedding = embedding_model.encode(
    [question]
)



# Retrieve the most relevant chunks


distances, indices = index.search(
    np.array(question_embedding).astype("float32"),
    k=2
)



# Build the context


retrieved_chunks = []

for index_number in indices[0]:

    retrieved_chunks.append(
        chunks[index_number]
    )


context = "\n\n".join(
    retrieved_chunks
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

Answer the user's question using ONLY the
provided context.

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



# Send context + question to Gemini


response = chain.invoke(
    {
        "context": context,
        "question": question
    }
)



# Display the final answer


print("\nAI Answer:")
print(response.content)

