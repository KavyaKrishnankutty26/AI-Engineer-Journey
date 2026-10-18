# AI Document Learning Assistant

## Project Overview

The AI Document Learning Assistant is a Retrieval-Augmented Generation (RAG) application that answers questions using information retrieved from a local text document.

It uses semantic search to find relevant document chunks and Google's Gemini model to generate answers based on the retrieved context.

## Features

- Loads a local text document.
- Splits text into smaller chunks.
- Generates embeddings using Sentence Transformers.
- Stores and searches embeddings using FAISS.
- Retrieves relevant chunks for a user's question.
- Generates context-based answers using Gemini.
- Handles questions when no relevant chunks are retrieved.
- Instructs the model not to invent information that is missing from the document.

## Technologies Used

- Python
- LangChain
- Google Gemini API
- Sentence Transformers
- FAISS
- NumPy
- python-dotenv

## Architecture

**Document → Text Chunks → Embeddings → FAISS Index → Relevant Chunks → Gemini → Answer**

## How It Works

1. Load the document.
2. Split the document into smaller chunks.
3. Convert each chunk into a numerical embedding.
4. Store the embeddings in a FAISS index.
5. Convert the user's question into an embedding.
6. Retrieve the most relevant chunks.
7. Send the context and question to Gemini.
8. Display the answer.

## Installation

### 1. Install dependencies

Run this command in your terminal:

```bash
pip install langchain langchain-community langchain-text-splitters langchain-google-genai sentence-transformers faiss-cpu numpy python-dotenv
```

### 2. Configure the API key

Create a `.env` file in the repository root:

```text
GEMINI_API_KEY=your_api_key_here
```

Replace the placeholder with your actual API key. Never commit the `.env` file to GitHub.

### 3. Prepare the document

Place `Document.txt` in the same folder as `day21_portfolio_rag.py`.

### 4. Run the application

From the `Week3/Day21` directory, run:

```bash
python day21_portfolio_rag.py
```

Enter a question about the document when prompted.

## Example Questions

- What is Python?
- How is Python useful in data engineering?
- Who invented Python?

The last question can help test whether the application avoids answering when the document doesn't contain the requested information.

## Current Limitations

- Currently supports a local text document.
- Processes the document and rebuilds the FAISS index when the application starts.
- Retrieval quality depends on chunking, embeddings, and the relevance threshold.
- The language model can still make mistakes despite grounding instructions.
- Gemini API usage is subject to available quotas.

## Future Improvements

- Support PDF and multiple document formats.
- Persist the FAISS index between runs.
- Add a user-friendly web interface.
- Improve retrieval evaluation.
- Add automated tests.

## Author

Kavya Krishnankutty

Aspiring AI Engineer | Data Engineering | Generative AI
