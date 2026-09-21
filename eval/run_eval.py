"""
NoteMind - retrieval evaluation harness (quantitative metrics).

Unlike tests/test_all.py (unit tests that check individual functions in
isolation, with fake data and fake embeddings), this script runs your
REAL question set against your REAL, already-ingested notes
(data/faiss.index + data/metadata.json) and reports actual numbers:

  - Hit@K       : fraction of questions where a "correct" chunk
                  (by keyword match, and optionally by source/page)
                  appears anywhere in the top-K retrieved results.
  - MRR         : Mean Reciprocal Rank -- rewards the correct chunk
                  appearing EARLIER in the results, not just present.
  - Avg latency : average wall-clock time per query, end to end
                  (embedding + FAISS search + retriever logic).

This requires you to have already run `python ingest.py` at least once,
so data/faiss.index and data/metadata.json exist. It also requires the
real embedding model (sentence-transformers), since retrieval quality
is exactly what's being measured -- this is NOT meant to run offline
with fake embeddings like the unit tests do.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --testset eval/qa_testset.json --top-k 5
    python eval/run_eval.py --save-results eval/results_2026-09-21.json

Extending it:
    Add more entries to eval/qa_testset.json. Each entry:
        {
          "id": "short_unique_name",
          "query": "the question to ask",
          "expected_keywords": ["keyword1", "keyword2"],  # ALL must
                                                            # appear
                                                            # (case-
                                                            # insensitive)
                                                            # in a hit
          "expected_source": "some_file.pdf",   # optional
          "expected_page": 12                    # optional
        }
    A retrieved chunk counts as "correct" for a question if:
      - every string in expected_keywords appears in the chunk's text
        (case-insensitive substring match), AND
      - expected_source matches the chunk's source, if given, AND
      - expected_page matches the chunk's page, if given.
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running this script directly (`python eval/run_eval.py`) as well
# as from the project root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.embeddings import create_query_embedding   # noqa: E402
from app.retriever import Retriever                  # noqa: E402
from app.storage import load_metadata                # noqa: E402
from app.vector_store import VectorStore              # noqa: E402


DEFAULT_INDEX_PATH = "data/faiss.index"
DEFAULT_METADATA_PATH = "data/metadata.json"
DEFAULT_TESTSET_PATH = "eval/qa_testset.json"
DEFAULT_TOP_K = 5


def load_testset(path):

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def result_matches_expectation(result, expectation):

    text = str(result.get("text", "")).lower()

    for keyword in expectation.get("expected_keywords", []):

        if keyword.lower() not in text:
            return False

    expected_source = expectation.get("expected_source")

    if expected_source and result.get("source") != expected_source:
        return False

    expected_page = expectation.get("expected_page")

    if expected_page is not None:

        try:
            if int(result.get("page", -1)) != int(expected_page):
                return False
        except (TypeError, ValueError):
            return False

    return True


def evaluate_one(retriever, expectation, top_k):

    query = expectation["query"]

    start = time.perf_counter()
    results = retriever.retrieve(query, top_k=top_k)
    elapsed = time.perf_counter() - start

    rank = None

    for position, result in enumerate(results, start=1):

        if result_matches_expectation(result, expectation):
            rank = position
            break

    hit = rank is not None
    reciprocal_rank = (1.0 / rank) if hit else 0.0

    return {
        "id": expectation.get("id", query),
        "query": query,
        "hit": hit,
        "rank": rank,
        "reciprocal_rank": reciprocal_rank,
        "latency_seconds": elapsed,
        "num_results_returned": len(results),
    }


def run_evaluation(
    testset_path,
    index_path,
    metadata_path,
    top_k,
):

    if not Path(index_path).exists() or not Path(metadata_path).exists():

        print(
            f"Could not find '{index_path}' and/or '{metadata_path}'.\n"
            "Run `python ingest.py` first to build the index from your "
            "notes, then re-run this evaluation."
        )

        sys.exit(1)

    testset = load_testset(testset_path)

    if not testset:

        print(f"No questions found in {testset_path}.")
        sys.exit(1)

    vector_store = VectorStore.load(index_path)
    metadata = load_metadata(metadata_path)
    retriever = Retriever(vector_store, metadata)

    per_question_results = []

    for expectation in testset:

        per_question_results.append(
            evaluate_one(retriever, expectation, top_k)
        )

    return per_question_results


def print_report(results, top_k):

    print()
    print("=" * 72)
    print(f"NoteMind Retrieval Evaluation  (top_k={top_k})")
    print("=" * 72)

    for result in results:

        status = "HIT " if result["hit"] else "MISS"
        rank_display = result["rank"] if result["hit"] else "-"

        print(
            f"[{status}] rank={rank_display:<3} "
            f"latency={result['latency_seconds']*1000:6.1f}ms  "
            f"{result['id']}: {result['query']!r}"
        )

    total = len(results)
    hits = sum(1 for r in results if r["hit"])
    hit_rate = hits / total if total else 0.0
    mrr = sum(r["reciprocal_rank"] for r in results) / total if total else 0.0
    avg_latency_ms = (
        sum(r["latency_seconds"] for r in results) / total * 1000
        if total
        else 0.0
    )

    print("-" * 72)
    print(f"Questions evaluated : {total}")
    print(f"Hit@{top_k}              : {hits}/{total}  ({hit_rate:.1%})")
    print(f"MRR                 : {mrr:.3f}")
    print(f"Avg latency         : {avg_latency_ms:.1f} ms/query")
    print("=" * 72)

    return {
        "top_k": top_k,
        "total_questions": total,
        "hits": hits,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "avg_latency_ms": avg_latency_ms,
        "per_question": results,
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run quantitative retrieval evaluation against real, "
            "already-ingested NoteMind data."
        )
    )

    parser.add_argument(
        "--testset",
        default=DEFAULT_TESTSET_PATH,
        help=f"Path to the JSON test set (default: {DEFAULT_TESTSET_PATH})",
    )
    parser.add_argument(
        "--index",
        default=DEFAULT_INDEX_PATH,
        help=f"Path to the FAISS index (default: {DEFAULT_INDEX_PATH})",
    )
    parser.add_argument(
        "--metadata",
        default=DEFAULT_METADATA_PATH,
        help=f"Path to metadata.json (default: {DEFAULT_METADATA_PATH})",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"How many results to retrieve per query (default: {DEFAULT_TOP_K})",
    )
    parser.add_argument(
        "--save-results",
        default=None,
        help="Optional path to write the full JSON report to.",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help=(
            "Optional hit-rate threshold (0-1). If given and the "
            "measured hit-rate is below it, the script exits with a "
            "non-zero status -- useful for CI."
        ),
    )

    args = parser.parse_args()

    results = run_evaluation(
        testset_path=args.testset,
        index_path=args.index,
        metadata_path=args.metadata,
        top_k=args.top_k,
    )

    report = print_report(results, args.top_k)

    if args.save_results:

        with open(args.save_results, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=2)

        print(f"\nFull report saved to {args.save_results}")

    if args.fail_under is not None and report["hit_rate"] < args.fail_under:

        print(
            f"\nHit-rate {report['hit_rate']:.1%} is below the "
            f"required {args.fail_under:.1%} threshold."
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
