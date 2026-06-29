import json
import re
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def clean_label(uri: str) -> str:
    """converts wn:tree.n.01 → tree"""
    # remove prefix e.g. wn:
    local = uri.split(":")[-1] if ":" in uri else uri
    # remove wordnet sense suffix e.g. tree.n.01 → tree
    local = re.sub(r"\.[a-z]\.\d+$", "", local)
    # underscores → spaces
    local = local.replace("_", " ").lower()
    return local.strip()


def relation_to_phrase(relation: str) -> str:
    """converts relation URIs to natural language phrases"""
    local = relation.split(":")[-1] if ":" in relation else relation.split("/")[-1]

    # explicit mapping for your KG's relation types
    relation_map = {
        "hasPart":  "has part",
        "hasColor": "has color",
        "in":       "is typically found in",
        "on":       "is typically found on",
        "near":     "is typically found near",
        "wears":    "typically wears",
        "holding":  "is typically holding",
        "behind":   "is typically behind",
    }

    if local in relation_map:
        return relation_map[local]

    # fallback: camelCase → words
    phrase = re.sub(r"(?<!^)(?=[A-Z])", " ", local).lower()
    return phrase.strip()

def triple_to_text(subject, relation, obj,
                   typicality, confidence, support, total) -> str:
    s = clean_label(subject)
    r = relation_to_phrase(relation)
    o = clean_label(obj)
    sentence = f"a {s} {r} {o}"
    meta = (f"[confidence: {confidence:.2f} · "
            f"typicality: {typicality:.2f} · "
            f"support: {support}/{total}]")
    return f"{sentence}  {meta}"


# ── RDF-star parser ───────────────────────────────────────────────────────────

def parse_rdfstar_ttl(ttl_path: str) -> list[dict]:
    """
    parses RDF-star turtle files with << subject predicate object >> syntax
    directly as text — no rdflib RDF-star dependency needed
    """

    # regex to match a full reified triple block:
    # << wn:tree.n.01 :hasPart wn:trunk.n.01 >>
    #     :typicality "0.97"^^xsd:float ;
    #     :confidence "0.84"^^xsd:float ;
    #     :support    "30"^^xsd:integer ;
    #     :total      "31"^^xsd:integer .
    pattern = re.compile(
        r'<<\s*(\S+)\s+(\S+)\s+(\S+)\s*>>'   # << s p o >>
        r'[^.]*?'                              # metadata lines
        r':typicality\s+"([^"]+)"'            # typicality value
        r'[^.]*?'
        r':confidence\s+"([^"]+)"'            # confidence value
        r'[^.]*?'
        r':support\s+"([^"]+)"'              # support value
        r'[^.]*?'
        r':total\s+"([^"]+)"',               # total value
        re.DOTALL
    )

    with open(ttl_path, "r", encoding="utf-8") as f:
        content = f.read()

    matches = pattern.findall(content)
    print(f"found {len(matches)} triples in {ttl_path}")

    results = []
    for subject, relation, obj, typicality, confidence, support, total in matches:
        try:
            t = float(typicality)
            c = float(confidence)
            s = int(support)
            tot = int(total)
        except ValueError:
            continue

        text = triple_to_text(subject, relation, obj, t, c, s, tot)
        results.append({
            "text":       text,
            "subject":    subject,
            "relation":   relation,
            "object":     obj,
            "typicality": t,
            "confidence": c,
            "support":    s,
            "total":      tot,
        })

    return results


# ── main serialiser ───────────────────────────────────────────────────────────

def serialise(ttl_path: str, output_path: str) -> list[dict]:
    print(f"parsing {ttl_path} ...")

    results = parse_rdfstar_ttl(ttl_path)

    if not results:
        print("no triples found — check your .ttl file format")
        return []

    # save to json
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"serialised {len(results)} triples → {output_path}")
    return results


# ── run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    results = serialise(
        ttl_path="C:/Users/singh/OneDrive/Desktop/kg_rag/data/facts_star_synset_02.ttl",
        output_path="C:/Users/singh/OneDrive/Desktop/kg_rag/data/processed/triples_text_02.json"
    )
    # preview first 5
    print("\n--- preview ---")
    for r in results[:5]:
        print(r["text"])
        
        