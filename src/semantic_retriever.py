import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────

CHROMA_PATH     = "chroma_db"
COLLECTION_NAME = "kg_triples"
MODEL_NAME      = "all-MiniLM-L6-v2"


# ── main retriever class ──────────────────────────────────────────────────────

class SemanticRetriever:

    def __init__(self):
        self.model      = SentenceTransformer(MODEL_NAME)
        self.client     = chromadb.PersistentClient(path=CHROMA_PATH)
        self.collection = self.client.get_collection(COLLECTION_NAME)
        print(f"semantic retriever ready — "
              f"{self.collection.count()} vectors loaded")

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """
        embeds the query and returns top_k most similar triples
        from ChromaDB using cosine similarity
        """
        # embed the query
        query_embedding = self.model.encode([query]).tolist()

        # search ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )

        # parse into clean triple dicts
        triples = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, distance in zip(documents, metadatas, distances):
            # chromadb cosine distance → similarity score (1 = perfect match)
            similarity = round(1 - distance, 4)

            triples.append({
                "text":       doc,
                "subject":    meta.get("subject",    ""),
                "relation":   meta.get("relation",   ""),
                "object":     meta.get("object",     ""),
                "typicality": meta.get("typicality", 0.0),
                "confidence": meta.get("confidence", 0.0),
                "support":    meta.get("support",    0),
                "total":      meta.get("total",      0),
                "similarity": similarity,
                "source":     "semantic",
            })

        return triples


# ── run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    retriever = SemanticRetriever()

    test_queries = [
        "what color is the sky?",
        "what parts does a tree have?",
        "what is a dog typically near?",
        "what does a woman wear?",
    ]

    for q in test_queries:
        print(f"\n{'='*50}")
        print(f"query: {q}")
        print(f"{'='*50}")
        results = retriever.retrieve(q, top_k=3)
        for r in results:
            print(f"  [{r['similarity']:.2f}] {r['text']}")