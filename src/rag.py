import json
from pathlib import Path
import re
from typing import List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import requests

#config

artifacts_dir = Path("data/artifacts")
chunks_file = artifacts_dir / "chunks.json"
index_file = artifacts_dir / "faiss.index"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

top_k=5

#data loaders

def load_chunks():
    with open(chunks_file, "r", encoding="utf-8") as f:
        return json.load(f)
    
def load_index():
    return faiss.read_index(str(index_file))

#retrieval logic

def retrieve(query: str, top_k: int = top_k)->List[str]:
    model= SentenceTransformer(embedding_model_name)
    index= load_index()
    chunks= load_chunks()

    query_embedding= model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    scores,indices= index.search(query_embedding, top_k)
    retrieved_texts=[]

    for idx in indices[0]:
        retrieved_texts.append(chunks[idx]["text"])
    return retrieved_texts


#prompt construction

def construct_prompt(query: str, retrieved_texts: List[str]) -> str:
    context = "\n\n".join(retrieved_texts)
    prompt = f"""
    You are an academic assistant.

    Answer the question strictly using the provided context.
    If the answer is not contained in the context, say:
    "I do not have enough information to answer this."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    return prompt.strip()



#LLM interface
#using llama3:8b via local REST API

def generate_answer(prompt: str) -> str:
    url="http://localhost:11434/api/generate"
    payload = {
        "model": "llama3:8b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9,
            "num_predict": 512
        }
    }
    response = requests.post(url, json=payload)
    response.raise_for_status()
    result = response.json()
    return result["response"].strip()

def answer_query(query: str) -> str:
    print(f"\nQuery: {query}\n")
    context_chunks=retrieve(query)
    prompt= construct_prompt(query, context_chunks)
    answer= generate_answer(prompt)

    print("Answer:\n")
    print(answer)



#command line execution entry point
if __name__ == "__main__":
    query = input("Enter your query: ").strip()
    answer_query(query)
