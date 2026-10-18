"""
AI Document Learning Assistant — RAG Application

Purpose:
Answer user questions using information retrieved
from a supplied text document.

Main components:
- Document loading and text chunking
- Embedding generation using Sentence Transformers
- Vector search using FAISS
- Answer generation using Google Gemini
- Basic error handling and source display
"""

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



# Load environment variables

load_dotenv()



# Load documents

def load_documents(file_path):
    try:
        loader = TextLoader(
          str(file_path),
          encoding="utf-8"
          )
        documents = loader.load()

        return documents

    except FileNotFoundError:
    
        print("Error : Document file was not found.")
        print("Please check the file file name and location")
        return None


# Create chunks


def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=40
    )

    chunks = splitter.split_documents(
        documents
    )

    return chunks



# Create embeddings

def create_embeddings(chunks):

    embedding_model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    chunk_texts = [
        chunk.page_content
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        chunk_texts
    )

    return embedding_model, embeddings



# Create FAISS index


def create_vector_store(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        np.array(embeddings).astype("float32")
    )

    return index



# Retrieve relevant documents


def retrieve_documents(
    question,
    embedding_model,
    index,
    chunks
):

    question_embedding = embedding_model.encode(
        [question]
    )

    distances, indices = index.search(
        np.array(question_embedding).astype("float32"),
        k=4
    )

    threshold = 1.0

    relevant_documents = []

    for distance, index_number in zip(
        distances[0],
        indices[0]
    ):

        if index_number<0:
            continue

        if distance <= threshold:

            document = chunks[index_number]

            relevant_documents.append(
                document
            )

    return relevant_documents



# Generate answer

def generate_answer(
    question,
    retrieved_documents,
    model
):

    # Build context with source information
    context_parts = []

    for document in retrieved_documents:

        source = document.metadata.get(
          "source",
          "Unknown source"
    )

    chunk_id = document.metadata.get(
        "chunk_id",
        "Unknown chunk"
    )

    context_parts.append(
        f"Source: {source}\n"
        f"Chunk ID: {chunk_id}\n"
        f"Content: {document.page_content}"
    )


    context = "\n\n".join(context_parts)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are a helpful AI assistant that answers questions
about documents.

Rules:
1. Answer using ONLY the provided context.
2. Do not use outside knowledge to fill missing details.
3. If the context does not contain enough information,
   say: "The information is not available in the document."
4. If only part of the answer is supported, explain
   what the context confirms and what is missing.
5. Do not invent facts, names, numbers, or explanations.
6. Keep the answer clear and directly related to
   the user's question.
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

    chain = prompt | model
    try:

        response = chain.invoke(
        {
            "context": context,
            "question": question
        }
        )

        return response

    except Exception as e:
         print("\nError:Unable to generate an AI answer")
         print("Please check your internet connectionp and try again later")

         return None
         print("\nTechnical Error",e)
         return None

#  Main application

def main():

    # Find the document
    file_path = Path(__file__).parent / "Document.txt"


    # Load documents
    documents = load_documents(
        file_path
    )
    if documents is None:
       return

    print(
        "Documents loaded:",
        len(documents)
    )

    if not any(
        document.page_content.strip()
        for document in documents
        ):
        print("Error, The document containing no text")
        return


    # Create chunks
    chunks = create_chunks(
        documents
    )

    print(
        "Chunks created:",
        len(chunks)
    )


    # Create embeddings
    embedding_model, embeddings = create_embeddings(
        chunks
    )

    print(
        "Embedding shape:",
        embeddings.shape
    )


    # Create FAISS vector store
    index = create_vector_store(
        embeddings
    )

    print(
        "Vectors stored in FAISS:",
        index.ntotal
    )


    # Create Gemini model
    model = ChatGoogleGenerativeAI(
        model="gemini-3.1-flash-lite"
    )


    # Get user's question
    question = input(
        "\nAsk a question about the document: "
    ).strip()


    # Validate question
    if not question:

        print("Please enter a question.")

        return


    # Retrieve relevant documents
    retrieved_documents = retrieve_documents(
        question,
        embedding_model,
        index,
        chunks
    )


    # Check retrieval result
    if not retrieved_documents:

        print(
            "\nThe information is not available "
            "in the document."
        )

        return


    # Generate final answer
    response = generate_answer(
        question,
        retrieved_documents,
        model
    )

    if response is None:
        return

    # Display answer
    print("\nAI Answer:")
    print(response.content)

    
# Display the sources used for retrieval
    print("\nSources used:")

    for document in retrieved_documents:

      source = document.metadata.get(
        "source",
        "Unknown source"
    )

    chunk_id = document.metadata.get(
        "chunk_id",
        "Unknown chunk"
    )

    print(f"- {source} (Chunk {chunk_id})")


# Run application

if __name__ == "__main__":
    main()

