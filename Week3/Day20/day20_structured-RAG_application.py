
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
# 2. Load documents
# --------------------------------------------------

def load_documents(file_path):

    loader = TextLoader(
        str(file_path),
        encoding="utf-8"
    )

    documents = loader.load()

    return documents


# --------------------------------------------------
# 3. Create chunks
# --------------------------------------------------

def create_chunks(documents):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=40
    )

    chunks = splitter.split_documents(
        documents
    )

    return chunks


# --------------------------------------------------
# 4. Create embeddings
# --------------------------------------------------

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


# --------------------------------------------------
# 5. Create FAISS index
# --------------------------------------------------

def create_vector_store(embeddings):

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        np.array(embeddings).astype("float32")
    )

    return index


# --------------------------------------------------
# 6. Retrieve relevant documents
# --------------------------------------------------

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

        if distance <= threshold:

            document = chunks[index_number]

            relevant_documents.append(
                document
            )

    return relevant_documents


# --------------------------------------------------
# 7. Generate answer
# --------------------------------------------------

def generate_answer(
    question,
    retrieved_documents,
    model
):

    context = "\n\n".join(
        document.page_content
        for document in retrieved_documents
    )

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

    chain = prompt | model

    response = chain.invoke(
        {
            "context": context,
            "question": question
        }
    )

    return response


# --------------------------------------------------
# 8. Main application
# --------------------------------------------------

def main():

    # Find the document
    file_path = Path(__file__).parent / "Document.txt"


    # Load documents
    documents = load_documents(
        file_path
    )

    print(
        "Documents loaded:",
        len(documents)
    )


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


    # Display answer
    print("\nAI Answer:")
    print(response.content)


# --------------------------------------------------
# 9. Run application
# --------------------------------------------------

if __name__ == "__main__":
    main()

