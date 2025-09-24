from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
sentence = ["The cat sits outside", "A dog plays in the park"]
embeddings = model.encode(sentence)
# print(embeddings)
# print(embeddings.shape)


# sementic search example
import numpy as np
query = "outdoor pets"

query_emb = model.encode([query])

cosine_sim =  np.dot(embeddings, query_emb.T)/ (
    np.linalg.norm(embeddings, axis=1, keepdims=True) * np.linalg.norm(query_emb, axis=1, keepdims=True)
)

# print(cosine_sim)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

X_embedded = TSNE(n_components=2, perplexity=1).fit_transform(embeddings)

plt.scatter(X_embedded[:,0], X_embedded[:,1])
for i, text in enumerate(sentence):
    plt.annotate(text, (X_embedded[i,0], X_embedded[i,1]))
plt.show()
