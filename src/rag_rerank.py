from calendar import c
import json
from multiprocessing import context
from pathlib import Path
import re
from typing import List,Dict,Any, final
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer,CrossEncoder
import requests

#config

artifacts_dir = Path("data/artifacts")
chunks_file = artifacts_dir / "chunks.json"
index_file = artifacts_dir / "faiss_index.index"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"


faiss_top_k=20
final_top_k=5


#data loaders

def load_chunks():
    with open(chunks_file, "r", encoding="utf-8") as f:
        return json.load(f)
    
def load_index():
    return faiss.read_index(str(index_file))

#model loaders
model = SentenceTransformer(embedding_model_name)
reranker= CrossEncoder(reranker_model_name)
index = load_index()
chunks = load_chunks()

#retrieval logic

def retrieve(query: str, top_k: int = final_top_k)->List[Dict[str,Any]]:

    query_embedding= model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)
    scores,indices= index.search(query_embedding, top_k)
    retrieved_texts=[]

    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        chunk = chunks[idx]
        retrieved_texts.append({
            "rank": rank,
            "score": float(score),
            "id": chunk.get("id"),
            "source_file": chunk.get("source_file"),
            "text": chunk.get("text", "")
        })
    return retrieved_texts

def rerank(query: str, retreived_chunks):
    pairs=[(query, ch["text"]) for ch in retreived_chunks]
    scores= reranker.predict(pairs)

    for ch, score in zip(retreived_chunks, scores):
        ch["rerank_score"]= float(score)
    
    reranked= sorted(retreived_chunks, key=lambda x: x["rerank_score"], reverse=True)

    return reranked

#prompt construction

def construct_prompt(query: str, retrieved_texts: List[Dict[str,Any]]) -> str:
    context_block=[]
    for ch in retrieved_texts:
        header = f"[Rank {ch['rank']} | Score {ch['score']:.4f} | Source {ch['source_file']} | Chunk {ch['id']}]"
        context_block.append(header + "\n" + ch["text"])

    context = "\n\n".join(context_block)
    prompt = f"""
You are an academic assistant.

Answer the question strictly using the provided context.
Focus only on aspects that directly answer the question.
RECHECK AND VERIFY FACTS before including them in your answer.

If the answer is not contained in the context, output EXACTLY:
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

def answer_query(query: str) -> None:
    print(f"\nQuery: {query}\n")

    #context_chunks = retrieve(query)
    candidate_chunks= retrieve(query, top_k=faiss_top_k)
    reranked= rerank(query, candidate_chunks)
    context_chunks = reranked[:final_top_k]
    prompt = construct_prompt(query, context_chunks)
    answer = generate_answer(prompt)

    print("Answer:\n")
    print(answer)

    print("\nSources:")
    seen_files = set()
    for ch in context_chunks:
        print(
        f"Rerank={ch['rerank_score']:.4f} | "
        f"FAISS rank={ch['rank']} | "
        f"{ch['source_file']} | {ch['id']}"
        )
        count = 0
        src = ch["source_file"]
        if src not in seen_files:
            seen_files.add(src)
            print(f"[Source: {src} | Chunk: {ch['id']}]")
            count += 1
        if count == 5:
            break




#command line execution entry point
if __name__ == "__main__":
    query = input("Enter your query: ").strip()
    answer_query(query)
