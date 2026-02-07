import os
import json
from pydoc import doc
import fitz  # PyMuPDF
from pathlib import Path
import re

#config

raw_data_dir="data/raw"
raw_data_dir=Path(raw_data_dir)
artifacts_dir="data/artifacts"
artifacts_dir = Path(artifacts_dir)
output_file=artifacts_dir/ "documents.json"

#text normalization
#remove artificial linebreaks
#Preserve sentence structure

def normalize_text(text: str) -> str:
    lines= text.splitlines()
    cleaned_lines=[]
    for line in lines:
        line=line.strip()
        if not line:
            cleaned_lines.append("")#para break
        else:
            cleaned_lines.append(line)
    text=" ".join(cleaned_lines)
    text=re.sub(r"\s+", " ", text) #normalize whitespace
    return text.strip()

#text extraction from PDF

def extract_text_from_pdf(file_path: Path) -> dict:
    doc=fitz.open(file_path)
    pages_text=[]
    for page in doc:
        page_text=page.get_text("text")
        if page_text:
            pages_text.append(page_text)
    full_text="\n".join(pages_text)
    full_text=normalize_text(full_text)

    return  {
        "source_file": file_path.name,
        "num_pages": len(doc),
        "text": full_text
    }


def ingest_pdfs()-> None:
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Raw data directory {raw_data_dir} does not exist.")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    documents=[]
    pdf_files=list(raw_data_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {raw_data_dir}.")
        return
    print(f"Found {len(pdf_files)} PDF files. Starting ingestion...")

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}...")
        try:
            doc_data=extract_text_from_pdf(pdf_file)
            if not doc_data["text"]:
                print(f"Warning: No text extracted from {pdf_file.name}. Skipping.")
                continue
            documents.append(doc_data)

        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")

    #write to json
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
    
    print(f"Ingestion complete. Processed {len(documents)} documents. Output written to {output_file}.")



    #Command line execution entry point

if __name__ == "__main__":
    ingest_pdfs()