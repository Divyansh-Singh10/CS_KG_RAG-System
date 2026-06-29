from src.sparql_retriever import SPARQLRetriever
from src.semantic_retriever import SemanticRetriever


# ── scoring config ────────────────────────────────────────────────────────────

# how much weight to give each signal when ranking merged results
# these add up to 1.0
CONFIDENCE_WEIGHT  = 0.50   # how statistically reliable the triple is
SIMILARITY_WEIGHT  = 0.30   # how semantically close to the query
TYPICALITY_WEIGHT  = 0.20   # how typical this relation is for the entity


# ── helpers ───────────────────────────────────────────────────────────────────

def compute_score(triple: dict) -> float:
    """
    computes a single ranking score for a triple
    combining confidence, semantic similarity and typicality
    """
    confidence  = triple.get("confidence",  0.0)
    similarity  = triple.get("similarity",  0.0)
    typicality  = triple.get("typicality",  0.0)

    return (
        CONFIDENCE_WEIGHT  * confidence  +
        SIMILARITY_WEIGHT  * similarity  +
        TYPICALITY_WEIGHT  * typicality
    )


def deduplicate(triples: list[dict]) -> list[dict]:
    seen = {}
    for triple in triples:
        # normalise key — extract local name only
        def local(uri):
            uri = str(uri)
            uri = uri.split("/")[-1].split("#")[-1].split(":")[-1]
            import re
            return re.sub(r"\.[a-z]\.\d+$", "", uri).lower()

        key = (
            local(triple.get("subject",  "")),
            local(triple.get("relation", "")),
            local(triple.get("object",   "")),
        )
        if key not in seen:
            seen[key] = triple
        else:
            if compute_score(triple) > compute_score(seen[key]):
                seen[key] = triple

    return list(seen.values())


def format_for_llm(triples: list[dict]) -> str:
    """
    formats the final ranked triples into a clean string
    ready to be injected into an LLM prompt as context
    """
    if not triples:
        return "No relevant facts found in the knowledge graph."

    lines = ["Retrieved facts from knowledge graph:\n"]
    for i, t in enumerate(triples, 1):
        confidence  = t.get("confidence",  0.0)
        typicality  = t.get("typicality",  0.0)
        support     = t.get("support",     0)
        total       = t.get("total",       0)
        score       = t.get("score",       0.0)

        # flag low confidence triples so LLM can hedge its answer
        certainty = "high" if confidence >= 0.4 else \
                    "medium" if confidence >= 0.15 else "low"

        lines.append(
            f"{i}. {t['text']}\n"
            f"   certainty: {certainty} | score: {score:.3f}"
        )

    return "\n".join(lines)


# ── main hybrid retriever class ───────────────────────────────────────────────

class HybridRetriever:

    def __init__(self):
        print("initialising hybrid retriever...")
        self.sparql   = SPARQLRetriever()
        self.semantic = SemanticRetriever()
        print("hybrid retriever ready")

    def retrieve(self, query: str,
                 top_k: int = 10,
                 sparql_k: int = 20,
                 semantic_k: int = 20) -> list[dict]:
        """
        main entry point — runs both retrievers, merges,
        deduplicates, scores and returns top_k ranked triples
        """

        # step 1 — run both retrievers in parallel
        print(f"\nretrieving for: '{query}'")
        sparql_results   = self.sparql.retrieve(query,   top_k=sparql_k)
        semantic_results = self.semantic.retrieve(query, top_k=semantic_k)

        print(f"  sparql:   {len(sparql_results)} results")
        print(f"  semantic: {len(semantic_results)} results")

        # step 2 — tag source and merge
        for t in sparql_results:
            t["source"]     = "sparql"
            t.setdefault("similarity", 0.0)   # sparql has no similarity score

        for t in semantic_results:
            t["source"]     = "semantic"
            t.setdefault("confidence", 0.0)   # fill in if missing

        merged = sparql_results + semantic_results

        # step 3 — deduplicate
        merged = deduplicate(merged)
        print(f"  after dedup: {len(merged)} unique triples")

        # step 4 — score and sort
        for t in merged:
            t["score"] = compute_score(t)

        merged.sort(key=lambda x: x["score"], reverse=True)

        # step 5 — return top_k
        return merged[:top_k]

    def retrieve_for_llm(self, query: str, top_k: int = 10) -> str:
        """
        convenience method — retrieves and formats in one call
        returns a string ready to paste into an LLM prompt
        """
        triples = self.retrieve(query, top_k=top_k)
        return format_for_llm(triples)


# ── run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    retriever = HybridRetriever()

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
        context = retriever.retrieve_for_llm(q, top_k=5)
        print(context)