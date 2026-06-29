import os
import re
from SPARQLWrapper import SPARQLWrapper, JSON
from dotenv import load_dotenv

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────

GRAPHDB_ENDPOINT = os.getenv(
    "GRAPHDB_ENDPOINT",
    "http://localhost:7200/repositories/kg_rag"
)

# ── helpers ───────────────────────────────────────────────────────────────────

# maps natural language question words to relation URIs in your KG
RELATION_MAP = {
    # hasColor triggers
    "color":   ":hasColor",
    "colour":  ":hasColor",
    "colored":  ":hasColor",
    "coloured": ":hasColor",

    # hasPart triggers
    "part":    ":hasPart",
    "parts":   ":hasPart",
    "made of": ":hasPart",
    "consist": ":hasPart",
    "contain": ":hasPart",

    # spatial / scene triggers
    "found in":  ":in",
    "located in": ":in",
    "seen in":   ":in",
    "in":        ":in",
    "on":        ":on",
    "near":      ":near",
    "behind":    ":behind",

    # action triggers
    "wear":    ":wears",
    "wearing": ":wears",
    "hold":    ":holding",
    "holding": ":holding",
}

# wordnet namespace prefix used in your KG
WN_PREFIX = "wn"


def extract_entity(query: str) -> str:
    """
    extracts the main entity from a natural language query
    e.g. "what color is the sky?" → "sky"
    """
    stopwords = [
        "what", "which", "where", "how", "does", "do", "is", "are",
        "the", "a", "an", "have", "has", "color", "colour", "part",
        "parts", "of", "typically", "found", "near", "on", "in",
        "behind", "wearing", "holding", "tell", "me", "about"
    ]
    # strip punctuation first
    clean = re.sub(r"[^\w\s]", "", query.lower())
    words = clean.split()
    keywords = [w for w in words if w not in stopwords]
    return keywords[0] if keywords else ""


def detect_relation(query: str) -> str | None:
    """
    detects which relation the query is asking about
    e.g. "what color is the sky?" → ":hasColor"
         "what parts does a tree have?" → ":hasPart"
    returns None if no relation detected (broad search)
    """
    q = query.lower()
    for keyword, relation in RELATION_MAP.items():
        if keyword in q:
            return relation
    return None


def entity_to_uri(entity: str) -> list[str]:
    """
    converts a plain entity name to possible wordnet URI candidates
    e.g. "sky" → ["wn:sky.n.01", "wn:sky.n.02"]
    we try the most common sense (.n.01) first, then broader search
    """
    # normalise: spaces → underscores, lowercase
    normalised = entity.strip().lower().replace(" ", "_")
    # return top 3 sense candidates
    return [
        f"wn:{normalised}.n.01",
        f"wn:{normalised}.n.02",
        f"wn:{normalised}.n.03",
    ]


# ── SPARQL queries ────────────────────────────────────────────────────────────

def query_by_entity_and_relation(entity_uri: str, relation: str) -> list[dict]:
    """
    precise lookup: given entity + relation, return all matching triples
    e.g. wn:sky.n.01 + :hasColor → all color triples for sky
    """
    sparql = SPARQLWrapper(GRAPHDB_ENDPOINT)
    sparql.setReturnFormat(JSON)

    query = f"""
    PREFIX wn: <http://example.org/wn/>
    PREFIX : <http://example.org/onto#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT ?object ?typicality ?confidence ?support ?total WHERE {{
        << {entity_uri} {relation} ?object >>
            :typicality ?typicality ;
            :confidence ?confidence ;
            :support    ?support ;
            :total      ?total .
    }}
    ORDER BY DESC(?confidence)
    """

    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
        return _parse_results(
            results,
            subject=entity_uri,
            relation=relation
        )
    except Exception as e:
        print(f"SPARQL error: {e}")
        return []


def query_by_entity_broad(entity_uri: str) -> list[dict]:
    """
    broad lookup: return ALL triples for an entity regardless of relation
    useful when no specific relation is detected in the query
    """
    sparql = SPARQLWrapper(GRAPHDB_ENDPOINT)
    sparql.setReturnFormat(JSON)

    query = f"""
    PREFIX wn: <http://example.org/wn/>
    PREFIX : <http://example.org/onto#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT ?relation ?object ?typicality ?confidence ?support ?total WHERE {{
        << {entity_uri} ?relation ?object >>
            :typicality ?typicality ;
            :confidence ?confidence ;
            :support    ?support ;
            :total      ?total .
    }}
    ORDER BY DESC(?confidence)
    LIMIT 20
    """

    sparql.setQuery(query)
    try:
        results = sparql.query().convert()
        return _parse_results(
            results,
            subject=entity_uri,
            relation=None
        )
    except Exception as e:
        print(f"SPARQL error: {e}")
        return []


def _parse_results(results: dict, subject: str, relation: str | None) -> list[dict]:
    """
    converts raw SPARQL JSON results into clean triple dicts
    matching the same format as the semantic retriever
    """
    triples = []
    bindings = results.get("results", {}).get("bindings", [])

    for row in bindings:
        obj        = row.get("object",     {}).get("value", "")
        typicality = float(row.get("typicality", {}).get("value", 0))
        confidence = float(row.get("confidence", {}).get("value", 0))
        support    = int(row.get("support",    {}).get("value", 0))
        total      = int(row.get("total",      {}).get("value", 0))
        rel        = relation or row.get("relation", {}).get("value", "")

        # convert URI back to short form for display
        obj_short = obj.split("/")[-1] if "/" in obj else obj
        rel_short = rel.split("/")[-1] if "/" in rel else rel

        # build natural language text (same format as serialiser)
        subj_label = subject.split(":")[-1]
        subj_label = re.sub(r"\.[a-z]\.\d+$", "", subj_label).replace("_", " ")
        obj_label  = re.sub(r"\.[a-z]\.\d+$", "", obj_short).replace("_", " ")

        text = (
            f"{subj_label} {rel_short} {obj_label}  "
            f"[confidence: {confidence:.2f} · "
            f"typicality: {typicality:.2f} · "
            f"support: {support}/{total}]"
        )

        triples.append({
            "text":       text,
            "subject":    subject,
            "relation":   rel,
            "object":     obj,
            "typicality": typicality,
            "confidence": confidence,
            "support":    support,
            "total":      total,
            "source":     "sparql",
        })

    return triples


# ── main retriever class ──────────────────────────────────────────────────────

class SPARQLRetriever:

    def __init__(self):
        self.endpoint = GRAPHDB_ENDPOINT

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """
        main entry point — takes a natural language query,
        extracts entity + relation, queries GraphDB, returns triples
        """
        entity   = extract_entity(query)
        relation = detect_relation(query)

        if not entity:
            print("could not extract entity from query")
            return []

        print(f"entity: '{entity}' | relation: '{relation}'")

        # try entity URIs from most to least specific sense
        candidates = entity_to_uri(entity)
        results = []

        for uri in candidates:
            if relation:
                results = query_by_entity_and_relation(uri, relation)
            else:
                results = query_by_entity_broad(uri)

            if results:
                print(f"found {len(results)} results for {uri}")
                break   # stop at first URI that returns results

        if not results:
            print(f"no SPARQL results found for '{entity}'")

        return results[:top_k]


# ── run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    retriever = SPARQLRetriever()

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
        results = retriever.retrieve(q)
        for r in results[:3]:
            print(f"  {r['text']}")