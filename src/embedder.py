import json
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# ── config ────────────────────────────────────────────────────────────────────

CHROMA_PATH      = "chroma_db"
COLLECTION_NAME  = "kg_triples"
MODEL_NAME       = "all-MiniLM-L6-v2"
BATCH_SIZE       = 128


# ── main embedder ─────────────────────────────────────────────────────────────

def embed_triples(json_path: str) -> None:
    """
    loads triples from json, embeds each text sentence,
    and stores vectors + metadata in ChromaDB
    """

    # load triples
    print(f"loading triples from {json_path} ...")
    with open(json_path, "r", encoding="utf-8") as f:
        triples = json.load(f)
    print(f"loaded {len(triples)} triples")

    # load embedding model
    print(f"loading embedding model: {MODEL_NAME} ...")
    model = SentenceTransformer(MODEL_NAME)

    # set up ChromaDB
    print(f"setting up ChromaDB at {CHROMA_PATH} ...")
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    # delete collection if it already exists (clean re-run)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"deleted existing collection: {COLLECTION_NAME}")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}   # cosine similarity
    )

    # embed and store in batches
    print(f"embedding {len(triples)} triples in batches of {BATCH_SIZE} ...")
    total_batches = (len(triples) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(0, len(triples), BATCH_SIZE), total=total_batches):
        batch = triples[i : i + BATCH_SIZE]

        texts      = [t["text"]     for t in batch]
        ids        = [f"triple_{i + j}" for j in range(len(batch))]
        metadatas  = [
            {
                "subject":    t["subject"],
                "relation":   t["relation"],
                "object":     t["object"],
                "typicality": t["typicality"],
                "confidence": t["confidence"],
                "support":    t["support"],
                "total":      t["total"],
            }
            for t in batch
        ]

        # generate embeddings
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        # store in ChromaDB
        collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    print(f"\ndone — {collection.count()} vectors stored in ChromaDB")


# ── quick sanity check ────────────────────────────────────────────────────────

def test_retrieval(query: str = "what color is the sky?") -> None:
    """
    runs a quick similarity search to confirm everything works
    """
    print(f"\ntesting retrieval: '{query}'")

    model  = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5
    )

    print("\ntop 5 results:")
    for doc, meta in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        print(f"  {doc}")


# ── run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    embed_triples("C:/Users/singh/OneDrive/Desktop/kg_rag/data/processed/triples_text_02.json")
    test_retrieval("what color is the sky?")