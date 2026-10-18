# AI Document Learning Assistant — RAG Application

## Project Overview

The AI Document Learning Assistant is a Retrieval-Augmented Generation (RAG) application built with Python, LangChain, FAISS, Sentence Transformers, and Google Gemini.

It allows users to ask questions about a text document and receive answers based on relevant information retrieved from that document.

## Features

- Loads a text document using LangChain.
- Splits the document into smaller chunks.
- Converts chunks into vector embeddings using Sentence Transformers.
- Stores embeddings in a FAISS vector index.
- Retrieves relevant chunks based on the user's question.
- Generates answers using Google Gemini.
- Displays the retrieved sources used by the application.
- Includes basic input validation and error handling.

## Technology Stack

- Python
- LangChain
- Google Gemini
- Sentence Transformers
- FAISS
- NumPy
- python-dotenv

## Project Structure

```text
Day20/
├── day20_structured_RAG_application.py
├── Document.txt
├── requirements.txt
└── README.md
```

## Setup Instructions

### 1. Install Python

Install a compatible Python version and verify your installation:

```bash
python --version
```

### 2. Install dependencies

From the project folder, run:

```bash
pip install -r requirements.txt
```

### 3. Configure your Gemini API key

Create a `.env` file in the project folder containing:

```text
GEMINI_API_KEY=your_gemini_api_key
```

Replace the placeholder with your own API key.

Never commit your `.env` file or publish your API key.

### 4. Prepare the document

Place the text you want the assistant to search in `Document.txt`.

### 5. Run the application

From the project folder, run:

```bash
python day20_structured_RAG_application.py
```

Enter a question when prompted.

## How It Works

1. **Document loading:** Reads the source text.
2. **Chunking:** Splits the document into smaller sections.
3. **Embedding:** Converts each chunk into a numerical vector.
4. **Vector indexing:** Stores the vectors in FAISS.
5. **Retrieval:** Finds chunks related to the user's question.
6. **Generation:** Sends the retrieved context and question to Gemini.
7. **Source display:** Shows the source information for retrieved chunks.

## Limitations

- Answers depend on the contents and quality of the supplied document.
- Retrieval quality depends on chunking, embeddings, and search settings.
- The application requires a valid Gemini API key and an internet connection for answer generation.
- The current version processes a text document rather than a collection of different document formats.

## Future Improvements

- Support PDF and multiple document formats.
- Improve retrieval using evaluation data.
- Add a user-friendly web interface.
- Add automated tests.
- Deploy the application.

## Security

Keep API keys and other secrets out of source control. Store them in environment variables and ensure `.env` is excluded by `.gitignore`.

## Author

Kavya Krishnankutty

Aspiring AI Engineer | Data Engineering | Generative AI
