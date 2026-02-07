import json
from pathlib import Path
from typing import List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


#config

artifacts_dir = Path("data/artifacts")

chunks_file = artifacts_dir/ "chunks.json"
metadata_file = artifacts_dir/ "metadata.json"
index_file = artifacts_dir/ "faiss.index"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

top_k = 5


#helper functions

def load_chunks():
    with open(chunks_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_metadata():
    with open(metadata_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_index():
    return faiss.read_index(str(index_file))


#retrivel logic

def retrieve(query: str, top_k: int = top_k):
    print(f"\nQuery: {query}\n")

    print("Loading model and index...")
    model = SentenceTransformer(embedding_model_name)
    index = load_index()

    chunks = load_chunks()

    # Embed query
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True
    )

    #normalize for cosine similarity
    faiss.normalize_L2(query_embedding)

    #search
    scores, indices = index.search(query_embedding, top_k)

    print("Top results:\n")

    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        chunk = chunks[idx]

        print(f"Rank {rank}")
        print(f"Score: {score:.4f}")
        print(f"Chunk ID: {chunk['id']}")
        print(f"Source: {chunk['source_file']}")
        print("Text:")
        print(chunk["text"][:800])  #prevent flooding terminal


#command line execution entry point

if __name__ == "__main__":
    query = input("Enter your query: ").strip()
    retrieve(query)
