
from pathlib import Path

from langchain_community.document_loaders import TextLoader



# Find the document

file_path = Path(__file__).parent / "Document.txt"



# Create the document loader


loader = TextLoader(
    str(file_path),
    encoding="utf-8"
)



# Load the document


documents = loader.load()
#print("\nFirst Document:")
#print(documents[0])

#print("\nFirst Document content:")
#print(documents[0].page_content)

#print("\nFirst Document metadata:")
#print(documents[0].metadata)



# Display the loaded documents

print("Number of documents:", len(documents))


for document in documents:

    print("\n--------------------")

    print("Page content:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)

