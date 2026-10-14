import os
from pathlib import Path
import numpy as np
from pathlib import Path
import faiss
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
#from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI

env_path = Path(__file__).resolve().parents[2]/".env"
load_dotenv(env_path)


def load_documents(file_path):

    loader = TextLoader(str(file_path),
    encoding="utf-8"
    ) 
    documents= loader.load()
    return documents

def create_chunks(documents):
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=40
    )
    chunks  = splitter.split_documents(documents)
    return chunks

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
    return embedding_model,embeddings
    
def create_vector_store(embeddings):

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) 
    index.add(
        np.array(embeddings).astype("float32")
    )
    return index

def retrieve_documents(
        question,
        embedding_model,
        index,
        chunks
):
    question_embedding = embedding_model.encode(
        [question]
    )
    distances,indices = index.search(
        np.array(question_embedding).astype("float32"),
        k=4
    )

    threshold = 1.0
    relevant_documents = []
    for distance,index_number in zip(
        distances[0],
        indices[0]
    ):

        if distance<=threshold:
            document = chunks[index_number]
            relevant_documents.append(
            document
        )

    return relevant_documents


def generate_answer(question,retrieved_documents,model):

   context = "\n\n".join(document.page_content
   for document in retrieved_documents
   )

   prompt = ChatPromptTemplate.from_messages(
      [
         (
            "system",
            """
            You are a helpful AI Assistant. answer using only the supplied context. 
            If the answer is not in the context, say:
              "The information is not available in the document".
              Do not invent information
        """
         ),
         (
            "human",
            """
            Context:
            {context}

            Question:
            {question}

            Explain the answer in simple language.
            """
         )
      ]
   )

   chain = prompt | model

   response = chain.invoke(
      {
         "context":context,
         "question":question
      }
   )

   return response

def main():

   file_path = Path(__file__).parent / "Document.txt"
   
   documents = load_documents(file_path)
   print("\n[1/4] Loading document...")
   print("Documents loaded",len(documents))
   
   chunks = create_chunks(documents)
   print("\n2/4 Creating chunks...")
   print("Chunks created",len(chunks))

   embedding_model,embeddings = create_embeddings(chunks) 
   print("\n[3/4] Creating embeddings...")
   print("Embedding shape",embeddings.shape)

   index = create_vector_store(embeddings)
   print("\n[4/4] Building FAISS index...")
   print("Vectors stores in FAISS",index.ntotal)

   question = input("\n Ask a question about the document").strip()
   if not question:
      print("Please enter a question")
      return
   retrieved_documents = retrieve_documents(
      question,
      embedding_model,
      index,
      chunks
   )

   if not retrieved_documents:
      print("Information is not available in the document")
      return
   print("\nRetrieved chunks")
   for document in retrieved_documents:
       print("-",document.page_content)

   print("\nNumber of retrieved chunks",len(retrieved_documents))

   model =ChatGoogleGenerativeAI(
   model = "gemini-3.1-flash-lite"
)
   response = generate_answer(question,retrieved_documents,model)
   print("\n===AI Document Learning Assistant===")
   print("\nAI Answer:")

   if isinstance(response.content,str):
       print(response.content)
   else:
       for item in response.content:
           if isinstance(item,dict) and item.get("type")=="text" :
                print(item.get("text",""))

if __name__ == "__main__":
       main()