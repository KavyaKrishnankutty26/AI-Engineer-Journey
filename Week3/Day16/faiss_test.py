
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np



# Load the embedding model


model = SentenceTransformer("all-MiniLM-L6-v2")



# Create some example documents


texts = [
    "Python is widely used in data engineering.",
    "Python is popular for artificial intelligence and machine learning.",
    "LangChain is used to build applications powered by large language models.",
    "RAG allows AI applications to retrieve information from external documents."
]



# Convert the texts into embeddings


embeddings = model.encode(texts)

print("Number of texts:", len(texts))
print("Embedding shape:", embeddings.shape)



# Create a FAISS index


dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)



# Add embeddings to FAISS


index.add(
    np.array(embeddings).astype("float32")
)

print("Vectors stored in FAISS:", index.ntotal)



# Create a user question


question = "How is Python useful for working with data?"



# Convert the question into an embedding


question_embedding = model.encode(
    [question]
)



# Search FAISS


distances, indices = index.search(
    np.array(question_embedding).astype("float32"),
    k=2
)



# Display the results

print("\nQuestion:")
print(question)

print("\nMost relevant results:")

for i, index_number in enumerate(indices[0]):

    print(f"\nResult {i + 1}:")
    print(texts[index_number])

    print("Distance:", distances[0][i])
