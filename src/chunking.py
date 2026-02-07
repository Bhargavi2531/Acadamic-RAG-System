import json
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer

# Configuration
artifacts_dir = Path("data/artifacts")
documents_file = artifacts_dir / "documents.json"
output_file = artifacts_dir / "chunks.json"

tokenizer_name="bert-base-uncased"

chunk_size=512
chunk_overlap=50  #tokens


#Chunking logic

class Chunker:
    def __init__(self,tokenizer_name:str):
        self.tokenizer=AutoTokenizer.from_pretrained(tokenizer_name,use_fast=True)
    def chunk_text(self,text:str) ->List[str]:
        encoding = self.tokenizer(
        text,
        max_length=chunk_size,
        stride=chunk_overlap,
        truncation=True,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        add_special_tokens=False
        )

        chunks = []

        for offsets in encoding["offset_mapping"]:
        # offsets is a list of (char_start, char_end) for one chunk
            char_start = offsets[0][0]
            char_end = offsets[-1][1]

            chunk = text[char_start:char_end].strip()
            if chunk:
                chunks.append(chunk)

        return chunks

    

def load_documents() ->List[Dict]:
    if not documents_file.exists():
        raise FileNotFoundError(f"run ingest.py first to create {documents_file}")
    with open(documents_file,"r",encoding="utf-8") as f:
        return json.load(f)
    
def save_chunks(chunks:List[Dict]) ->None:
    with open(output_file,"w",encoding="utf-8") as f:
        json.dump(chunks,f,ensure_ascii=False,indent=2)

def chunk_documents() ->None:
    documents=load_documents()
    chunker=Chunker(tokenizer_name)

    all_chunks=[]
    chunk_id=0

    print(f"Chunking {len(documents)} documents...")
    print("\nStarting chunking process...")

    for doc in documents:
        text=doc["text"]
        source_file=doc["source_file"]

        text_chunks=chunker.chunk_text(text)

        for ch in text_chunks:
            all_chunks.append({
                "id": f"chunk_{chunk_id:06d}",
                "text": ch,
                "source_file": source_file,
            })
            chunk_id+=1
    
    save_chunks(all_chunks)
    print(f"Chunking complete. {len(all_chunks)} chunks saved to {output_file}.")



#Command line execution entry point

if __name__=="__main__":
    chunk_documents()

