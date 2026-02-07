import numpy as np
import faiss
from pathlib import Path

#config
artifacts_dir= Path("data/artifacts")
embeddings_file= artifacts_dir / "embeddings.npy"
index_file= artifacts_dir / "faiss_index.index"


#indexing

def build_index() -> None:
    if not embeddings_file.exists():
        raise FileNotFoundError(
            f"{embeddings_file} not found. Run embed.py first."
        )

    print("Loading embeddings...")
    embeddings = np.load(embeddings_file)

    print(f"Embeddings shape: {embeddings.shape}")

    #normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"Total vectors in index: {index.ntotal}")

    faiss.write_index(index, str(index_file))

    print(f"FAISS index saved to {index_file}")


# Command-line execution entry point

if __name__ == "__main__":
    build_index()
