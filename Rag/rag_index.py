from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np
import math
import json
import os
from typing import List, Dict

EMB_MODEL = "all-MiniLM-L6-v2"   # small for demo; swap for larger if you need quality
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # optional reranker

embed_model = SentenceTransformer(EMB_MODEL)
# Optional: a cross-encoder for reranking
reranker = CrossEncoder(CROSS_ENCODER_MODEL)

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Simple whitespace chunking with overlap (characters)."""
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk_tokens = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk_tokens))
        i += chunk_size - chunk_overlap
    return chunks

def build_index(docs: List[Dict], index_path="faiss_index"):
    """
    docs: List of dicts {'id': str, 'text': str, 'meta': {...}}
    Will create FAISS index + metadata store (json).
    """
    # 1) chunk docs and prepare passages
    passages = []
    for doc in docs:
        chunks = chunk_text(doc["text"], chunk_size=150, chunk_overlap=30)
        for i, c in enumerate(chunks):
            pid = f"{doc['id']}_chunk_{i}"
            passages.append({
                "id": pid,
                "text": c,
                "meta": {"source_id": doc["id"], **doc.get("meta", {})}
            })

    # 2) embed passages in batches
    texts = [p["text"] for p in passages]
    embeddings = embed_model.encode(texts, batch_size=64, convert_to_numpy=True, show_progress_bar=True).astype("float32")

    # 3) build FAISS index
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)  # use inner product for cosine (embedding normalized). You can use IndexIVFFlat for large corpora.
    # normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # 4) save index and metadata
    os.makedirs(index_path, exist_ok=True)
    faiss.write_index(index, os.path.join(index_path, "index.faiss"))
    # store passages metadata
    with open(os.path.join(index_path, "passages.json"), "w", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False, indent=2)

    print(f"Indexed {len(passages)} passages. dimension={d}")

# Example usage:
if __name__ == "__main__":
    docs = [
        {
            "id": "doc1",
            "text": """
    Artificial Intelligence (AI) refers to the simulation of human intelligence 
    in machines that are programmed to think, reason, and learn. AI includes 
    various subfields such as machine learning, natural language processing, 
    computer vision, and robotics. Modern AI applications range from 
    recommendation systems (like Netflix or Amazon), voice assistants 
    (Siri, Alexa), self-driving cars, to large language models such as GPT.

    Key aspects of AI:
    - Perception (computer vision, speech recognition)
    - Reasoning and decision making
    - Learning from data
    - Interaction through natural language
    """,
            "meta": {
                "title": "Introduction to Artificial Intelligence",
                "author": "AI Researcher",
                "tags": ["AI", "ML", "Intro"],
                "source": "AI Textbook"
            }
        },
        {
            "id": "doc2",
            "text": """
    PyTorch is an open-source deep learning framework developed by Facebook's AI Research lab. 
    It provides a flexible platform for building neural networks and supports dynamic computation 
    graphs, making it easy for researchers and practitioners to experiment with models.

    Key features of PyTorch:
    - Tensor operations with GPU acceleration
    - Autograd for automatic differentiation
    - Torch.nn for building neural networks
    - Torchvision for computer vision tasks
    - Strong community and ecosystem support

    PyTorch is widely used in research as well as production, and forms the basis of 
    many modern AI systems, including models for NLP and computer vision.
    """,
            "meta": {
                "title": "PyTorch Basics",
                "author": "Deep Learning Engineer",
                "tags": ["PyTorch", "Deep Learning", "Framework"],
                "source": "PyTorch Documentation"
            }
        }
    ]

    build_index(docs, index_path="my_rag_index")
