"""
Embeds all MedDRA terms into ChromaDB using ChromaDB's built-in ONNX
`all-MiniLM-L6-v2` embedding (no PyTorch, avoids segfaults with torch 2.10).

Each document is "PT_NAME | synonym1 | synonym2 | ..." for broad semantic matching.
"""

import json
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    CHROMA_DB_PATH,
    CHROMA_COLLECTION_NAME,
    MEDDRA_TERMS_PATH,
)


def main():
    if not MEDDRA_TERMS_PATH.exists():
        print(f"ERROR: {MEDDRA_TERMS_PATH} not found. Run curate_meddra.py first.")
        sys.exit(1)

    with open(MEDDRA_TERMS_PATH) as f:
        terms = json.load(f)

    print(f"Loaded {len(terms)} MedDRA terms from {MEDDRA_TERMS_PATH}")
    print(f"ChromaDB path   : {CHROMA_DB_PATH}")

    # ----- Step 1: precompute embeddings with ONNX (no torch, no segfault) -----
    print("\n[1/3] Loading ONNX all-MiniLM-L6-v2 embedding function...")
    ef = embedding_functions.DefaultEmbeddingFunction()

    documents = [t["search_text"] for t in terms]
    print(f"[2/3] Encoding {len(documents)} documents...")
    embeddings = ef(documents)
    print(f"      Done. {len(embeddings)} vectors of dim {len(embeddings[0])}")

    # ----- Step 2: reset collection -----
    print("\n[3/3] Writing to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    existing = [c.name for c in client.list_collections()]
    if CHROMA_COLLECTION_NAME in existing:
        print(f"      Deleting existing collection '{CHROMA_COLLECTION_NAME}'...")
        client.delete_collection(CHROMA_COLLECTION_NAME)

    collection = client.create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # ----- Step 3: insert in small batches -----
    ids = [f"PT_{t['pt_code']}_{i}" for i, t in enumerate(terms)]
    metadatas = [
        {
            "pt_name": t["pt_name"],
            "pt_code": t["pt_code"],
            "soc_name": t["soc_name"],
            "hlt_name": t["hlt_name"],
        }
        for t in terms
    ]

    batch_size = 50
    total = len(ids)
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        collection.add(
            ids=ids[i:end],
            documents=documents[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
        )
        print(f"      Inserted {end}/{total}")

    count = collection.count()
    print(f"\n✓ ChromaDB collection '{CHROMA_COLLECTION_NAME}' now has {count} documents.")
    print("  Next: python scripts/test_rag.py --no-ollama")


if __name__ == "__main__":
    main()
