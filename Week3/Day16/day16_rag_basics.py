from pathlib import Path
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = Path(__file__).parent / "document.txt"

loader = TextLoader(
    str(file_path),
    encoding = "utf-8"
)

documents = loader.load()

print("Number of documents:",len(documents))

print("\nOriginal document:")
print(documents[0].page_content)

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 200,
    chunk_overlap = 50
)

chunks = splitter.split_documents(documents)

print("\nNumber of chunks:",len(chunks))

for i, chunk in enumerate(chunks,start = 1):
    print(f"\n--- Chunk {i} ---")
    print(chunk.page_content)

    print("Metadata:",chunk.metadata)