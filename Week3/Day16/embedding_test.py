from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text = "python is good for data engineering." 

embedding = model.encode(text)

print("Embedding")
print(embedding)

print("\nEmbedding Dimesions")
print(len(embedding))
