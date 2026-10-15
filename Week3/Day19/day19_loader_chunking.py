
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



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
    chunk_overlap=40
)



# Split the Documents into smaller Documents

chunks = splitter.split_documents(
    documents
)


print("Chunks created:", len(chunks))



# Display the chunks


for i, chunk in enumerate(chunks, start=1):

    print("\n--------------------")

    print("Chunk:", i)

    print("\nPage content:")
    print(chunk.page_content)

    print("\nMetadata:")
    print(chunk.metadata)

