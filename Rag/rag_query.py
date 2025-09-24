from sentence_transformers import SentenceTransformer
import faiss, json, os, numpy as np
from typing import List, Tuple

EMB_MODEL = "all-MiniLM-L6-v2"
embed_model = SentenceTransformer(EMB_MODEL)

INDEX_PATH = "my_rag_index"

# load
index = faiss.read_index(os.path.join(INDEX_PATH, "index.faiss"))
with open(os.path.join(INDEX_PATH, "passages.json"), "r", encoding="utf-8") as f:
    passages = json.load(f)   # list indexed in same order as embeddings

def retrieve(query: str, k: int = 5) -> List[Tuple[dict, float]]:
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q_emb)
    distances, idxs = index.search(q_emb, k)
    results = []
    for idx, dist in zip(idxs[0], distances[0]):
        results.append((passages[int(idx)], float(dist)))  # higher distance = more similar because of IP on normalized vectors
    return results

# optional reranking using cross-encoder (more expensive but accurate)
from sentence_transformers import CrossEncoder
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # optional

def rerank(query: str, retrieved: List[Tuple[dict, float]], top_n=5):
    pairs = [(query, r[0]["text"]) for r in retrieved]
    scores = reranker.predict(pairs)
    # attach new score
    ranked = sorted(zip(retrieved, scores), key=lambda x: x[1], reverse=True)
    return [(item[0], float(score)) for item, score in ranked][:top_n]

# assemble prompt (simple)
SYSTEM_PROMPT = "You are an assistant that answers questions based on the provided context. Use only the context."

def assemble_prompt(query: str, retrieved: List[Tuple[dict, float]], max_chars=2000) -> str:
    # include top passages until length limit
    ctx = ""
    for p, score in retrieved:
        snippet = p["text"]
        # keep adding until we approach max_chars
        if len(ctx) + len(snippet) > max_chars:
            break
        ctx += f"\nSource: {p['meta'].get('title', p['meta'].get('source_id'))}\n{snippet}\n"
    prompt = f"{SYSTEM_PROMPT}\n\nContext:{ctx}\n\nQuestion: {query}\nAnswer:"
    return prompt

# Example usage
if __name__ == "__main__":
    q = "provide Key aspects of AI"
    retrieved = retrieve(q, k=8)
    # optional: reranked = rerank(q, retrieved)
    prompt = assemble_prompt(q, retrieved[:4])
    print(prompt)
