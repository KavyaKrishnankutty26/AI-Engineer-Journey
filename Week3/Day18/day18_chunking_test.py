from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = Path(__file__).parent / "document.txt"
text = file_path.read_text(
    encoding= "utf-8"
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 20

)

chunks = splitter.split_text(text)

print("Number of chunks:",len(chunks))

for i, chunk in enumerate(chunks,start=1):
    print(f"\n---chunks {i}---")
    print(chunk)
    print("\ncharacters:", len(chunk))