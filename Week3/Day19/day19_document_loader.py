
from pathlib import Path

from langchain_community.document_loaders import TextLoader


# --------------------------------------------------
# 1. Find the document
# --------------------------------------------------

file_path = Path(__file__).parent / "Document.txt"


# --------------------------------------------------
# 2. Create the document loader
# --------------------------------------------------

loader = TextLoader(
    str(file_path),
    encoding="utf-8"
)


# --------------------------------------------------
# 3. Load the document
# --------------------------------------------------

documents = loader.load()
#print("\nFirst Document:")
#print(documents[0])

#print("\nFirst Document content:")
#print(documents[0].page_content)

#print("\nFirst Document metadata:")
#print(documents[0].metadata)


# --------------------------------------------------
# 4. Display the loaded documents
# --------------------------------------------------

print("Number of documents:", len(documents))


for document in documents:

    print("\n--------------------")

    print("Page content:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)

