from langchain_core.documents import Document

document = Document(
    page_content= "Python is widely used in data engineering.",
    metadata = {
        "source": "document.txt",
        "chunk_id": 1
    }
)

print("Document:")
print(document)

print("\npage content:")
print(document.page_content)

print("\nmetadata:")
print(document.metadata)