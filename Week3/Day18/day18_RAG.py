import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate



#  Load environment variables

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")



# Create embedding model

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)



# Load document

file_path = Path(__file__).parent / "document.txt"

text = file_path.read_text(
    encoding="utf-8"
)



# Create chunks


chunks = [
    chunk.strip()
    for chunk in text.split("\n\n")
    if chunk.strip()
]

print("Number of chunks:", len(chunks))



#  Create embeddings for chunks


chunk_embeddings = embedding_model.encode(
    chunks
)



# Create FAISS index


dimension = chunk_embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(
    np.array(chunk_embeddings).astype("float32")
)

print("Vectors stored in FAISS:", index.ntotal)



# Get user question


question = input(
    "\nAsk a question about the document: "
).strip()


if not question:
    print("Please enter a question.")
    exit()



# Convert question into embedding


question_embedding = embedding_model.encode(
    [question]
)



# Search FAISS


#  Retrieve 4 candidates instead of 2
distances, indices = index.search(
    np.array(question_embedding).astype("float32"),
    k=4
)



# Apply relevance threshold

#  Maximum distance we will accept
threshold = 1.0

retrieved_chunks = []


#  Check every retrieved candidate
for distance, index_number in zip(
    distances[0],
    indices[0]
):

    print(
        f"\nDistance: {distance:.2f}"
    )

    print(
        "Chunk:",
        chunks[index_number]
    )


    #  Keep the chunk only if
    # its distance is within the threshold
    if distance <= threshold:

        retrieved_chunks.append(
            chunks[index_number]
        )



# Display accepted chunks

print("\nRelevant chunks:")

if retrieved_chunks:

    for chunk in retrieved_chunks:
        print("\n---")
        print(chunk)

else:

    print("No sufficiently relevant chunks found.")



# Create context


context = "\n\n".join(
    retrieved_chunks
)



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



# Create chain


chain = prompt | model



# Send context + question to Gemini


response = chain.invoke(
    {
        "context": context,
        "question": question
    }
)



# Display final answer
print("\nAI Answer:")
print(response.content)

