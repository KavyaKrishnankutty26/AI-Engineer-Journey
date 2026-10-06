
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


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

load_dotenv()


# --------------------------------------------------
# 2. Find the document
# --------------------------------------------------

file_path = Path(__file__).parent / "Document.txt"


# --------------------------------------------------
# 3. Load the document
# --------------------------------------------------

loader = TextLoader(
    str(file_path),
    encoding="utf-8"
)

documents = loader.load()

print("Documents loaded:", len(documents))


# --------------------------------------------------
# 4. Create the text splitter
# --------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=20
)


# --------------------------------------------------
# 5. Split Documents into chunks
# --------------------------------------------------

chunks = splitter.split_documents(
    documents
)

print("Chunks created:", len(chunks))


# --------------------------------------------------
# 6. Create the embedding model
# --------------------------------------------------

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 7. Get text from every chunk
# --------------------------------------------------

chunk_texts = [
    chunk.page_content
    for chunk in chunks
]


# --------------------------------------------------
# 8. Create embeddings
# --------------------------------------------------

embeddings = embedding_model.encode(
    chunk_texts
)


# --------------------------------------------------
# 9. Create FAISS index
# --------------------------------------------------

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)


# --------------------------------------------------
# 10. Store embeddings
# --------------------------------------------------

index.add(
    np.array(embeddings).astype("float32")
)

print("Vectors stored in FAISS:", index.ntotal)


# --------------------------------------------------
# 11. Get user's question
# --------------------------------------------------

question = input(
    "\nAsk a question about the document: "
).strip()


# --------------------------------------------------
# 12. Validate question
# --------------------------------------------------

if not question:

    print("Please enter a question.")

    exit()


# --------------------------------------------------
# 13. Convert question into an embedding
# --------------------------------------------------

question_embedding = embedding_model.encode(
    [question]
)


# --------------------------------------------------
# 14. Search FAISS
# --------------------------------------------------

distances, indices = index.search(
    np.array(question_embedding).astype("float32"),
    k=2
)


# --------------------------------------------------
# 15. Collect retrieved Documents
# --------------------------------------------------

retrieved_documents = []

for distance, index_number in zip(
    distances[0],
    indices[0]
):

    document = chunks[index_number]

    retrieved_documents.append(document)


# --------------------------------------------------
# 16. Create context for Gemini
# --------------------------------------------------

context = "\n\n".join(
    document.page_content
    for document in retrieved_documents
)


print("\nRetrieved context:")
print(context)


# --------------------------------------------------
# 17. Create Gemini model
# --------------------------------------------------

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite"
)


# --------------------------------------------------
# 18. Create RAG prompt
# --------------------------------------------------

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


# --------------------------------------------------
# 19. Create the chain
# --------------------------------------------------

chain = prompt | model


# --------------------------------------------------
# 20. Ask Gemini
# --------------------------------------------------

response = chain.invoke(
    {
        "context": context,
        "question": question
    }
)


# --------------------------------------------------
# 21. Display the final answer
# --------------------------------------------------

print("\nAI Answer:")
print(response.content)

