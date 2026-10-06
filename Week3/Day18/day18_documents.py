from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = Path(__file__).parent/"document.txt"
text = file_path.read_text(
    encoding="utf-8"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)
print("Number of chunks:",len(chunks))

documents = []
for i,chunk in enumerate(chunks,start=1):
    document = Document(
        page_content=chunk,
        metadata={
            "source":"document.txt",
            "chunk_id":1
        }
    )

    documents.append(document)

    for document in documents:
        print("\n-----------")
        print("Type")
        print(type(document))
        print("Page content")
        print(document.page_content)

        print("\nmetadata:")
        print(document.metadata)