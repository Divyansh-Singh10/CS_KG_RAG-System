import sys
import os

# make sure src is importable from tests/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.hybrid_retriever import HybridRetriever

# ── test queries ──────────────────────────────────────────────────────────────

# covers all 6 relation types in your KG
TEST_QUERIES = [
    # hasColor
    {
        "query":    "what color is the sky?",
        "expected": "blue",
        "relation": "hasColor"
    },
    {
        "query":    "what color is grass?",
        "expected": "green",
        "relation": "hasColor"
    },
    # hasPart
    {
        "query":    "what parts does a tree have?",
        "expected": "trunk",
        "relation": "hasPart"
    },
    {
        "query":    "what parts does a building have?",
        "expected": "window",
        "relation": "hasPart"
    },
    # near
    {
        "query":    "what is a dog typically near?",
        "expected": "man",
        "relation": "near"
    },
    # wears
    {
        "query":    "what does a woman wear?",
        "expected": "dress",
        "relation": "wears"
    },
    # on
    {
        "query":    "what is a bird typically on?",
        "expected": "branch",
        "relation": "on"
    },
    # holding
    {
        "query":    "what does a hand hold?",
        "expected": "knife",
        "relation": "holding"
    },
]


# ── test runner ───────────────────────────────────────────────────────────────

def run_tests():
    print("=" * 60)
    print("WEEK 1 DELIVERABLE — retrieval_test.py")
    print("=" * 60)

    retriever = HybridRetriever()

    passed = 0
    failed = 0
    results_log = []

    for test in TEST_QUERIES:
        query    = test["query"]
        expected = test["expected"].lower()
        relation = test["relation"]

        print(f"\n{'─'*60}")
        print(f"query:    {query}")
        print(f"relation: {relation} | expected keyword: '{expected}'")

        triples = retriever.retrieve(query, top_k=5)

        if not triples:
            print(f"FAIL — no results returned")
            failed += 1
            results_log.append((query, "FAIL", "no results"))
            continue

        # check if expected keyword appears in any of the top 5 results
        top_texts = [t["text"].lower() for t in triples]
        found     = any(expected in text for text in top_texts)
        top_result = triples[0]["text"]
        top_score  = triples[0].get("score", 0.0)
        top_cert   = "high" if triples[0].get("confidence", 0) >= 0.4 else \
                     "medium" if triples[0].get("confidence", 0) >= 0.15 else "low"

        if found:
            print(f"PASS ✓")
            print(f"  top result: {top_result}")
            print(f"  score: {top_score:.3f} | certainty: {top_cert}")
            passed += 1
            results_log.append((query, "PASS", top_result))
        else:
            print(f"FAIL — '{expected}' not found in top 5 results")
            print(f"  top result: {top_result}")
            failed += 1
            results_log.append((query, "FAIL", top_result))

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"passed: {passed}/{len(TEST_QUERIES)}")
    print(f"failed: {failed}/{len(TEST_QUERIES)}")
    print(f"score:  {round(passed/len(TEST_QUERIES)*100)}%")

    print(f"\n{'─'*60}")
    for query, status, result in results_log:
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon} {query}")
        print(f"    → {result[:80]}")

    print(f"\n{'='*60}")
    if failed == 0:
        print("ALL TESTS PASSED — retrieval pipeline is working correctly")
        print("ready to move to Week 2")
    elif passed >= len(TEST_QUERIES) * 0.75:
        print("MOSTLY PASSING — retrieval pipeline is working well enough")
        print("minor gaps are expected given KG coverage — ready for Week 2")
    else:
        print("NEEDS ATTENTION — check failed queries before Week 2")
    print(f"{'='*60}")


# ── run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_tests()