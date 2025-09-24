from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Step 1: Create sentences
sentences = [
    "The cat sits outside",
    "A dog plays in the park",
    "I love eating pizza",
    "Artificial intelligence is powerful",
    "He is reading a book",
    "The weather is sunny today"
]

# Step 2: Generate embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(sentences).astype("float32")

# Step 3: Build FAISS index
d = embeddings.shape[1]   # dimension of embeddings
index = faiss.IndexFlatL2(d)   # L2 = Euclidean distance
index.add(embeddings)          # add sentence embeddings to index

print(f"Number of sentences in index: {index.ntotal}")

# Step 4: Query
query = "I enjoy outdoor activities with my dog"
query_emb = model.encode([query]).astype("float32")

# Step 5: Search top 2 most similar sentences
distances, indices = index.search(query_emb, k=2)

print("\nQuery:", query)
print("\nTop matches:")
for idx, dist in zip(indices[0], distances[0]):
    print(f"  - {sentences[idx]} (distance: {dist:.4f})")
