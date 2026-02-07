import json
from pathlib import Path
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer


#config

artifacts_dir = Path("data/artifacts")
chunks_file= artifacts_dir / "chunks.json"
embeddings_file= artifacts_dir / "embeddings.npy"
metadata_file= artifacts_dir / "metadata.json"
embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
batch_size=16

#helper functions

def load_chunks() ->List[Dict]:
    if not chunks_file.exists():
        raise FileNotFoundError(f"run chunking.py first to create {chunks_file}")
    with open(chunks_file,"r",encoding="utf-8") as f:
        return json.load(f)

def save_embeddings(embeddings:np.ndarray) ->None:
    np.save(embeddings_file,embeddings)

def save_metadata(metadata:List[Dict]) ->None:
    with open(metadata_file,"w",encoding="utf-8") as f:
        json.dump(metadata,f,ensure_ascii=False,indent=2)


#embedding

def embed_chunks() ->None:
    chunks=load_chunks()
    print(f"Loaded {len(chunks)} chunks for embedding.")
    model=SentenceTransformer(embedding_model_name)

    texts=[chunk["text"] for chunk in chunks]
    all_embeddings=[]

    print(f"Starting embedding with batch size {batch_size}...")

    for i in range(0,len(texts),batch_size):
        batch_size_texts=texts[i:i+batch_size]
        batch_embeddings=model.encode(batch_size_texts,show_progress_bar=False,convert_to_numpy=True)
        all_embeddings.append(batch_embeddings)

        print(f"embedded chunks {i} to {i+len(batch_size_texts)-1}")

    embeddings=np.vstack(all_embeddings)
    save_embeddings(embeddings)

    metadata=[
            {
                "id":chunk["id"],
                "source_file":chunk["source_file"]
            }
            for chunk in chunks
        ]

    save_metadata(metadata)
    print(f"Embedding complete. Embeddings saved to {embeddings_file} and metadata to {metadata_file}.")
    print(f"embeddings shape: {embeddings.shape}")
#COMMAND LINE INTERFACE entry point
if __name__=="__main__":
    embed_chunks()
