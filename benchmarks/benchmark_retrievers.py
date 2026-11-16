import csv
import sys
import time
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INGESTION_DIR = PROJECT_ROOT / "src" / "ingestion"
RETRIEVAL_DIR = PROJECT_ROOT / "src" / "retrieval"
POLICY_FILE = PROJECT_ROOT / "data" / "shop_policies.txt"
CSV_OUTPUT_PATH = Path(__file__).resolve().parent / "day6_retriever_benchmarks.csv"

for folder in (INGESTION_DIR, RETRIEVAL_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from loader import load_and_chunk_file  # type: ignore  # noqa: E402
from retriever import retrieve_top_chunks  # type: ignore  # noqa: E402
from vector_search import retrieve_semantic_chunks  # type: ignore  # noqa: E402
from hybrid_retriever import retrieve_hybrid_chunks  # type: ignore  # noqa: E402

BENCHMARK_SUITE: list[dict] = [
    {
        "id": "Q1",
        "category": "Keyword",
        "query": "laptop warranty motherboard battery cooling system",
        "expected_cards": {1, 2},
    },
    {
        "id": "Q2",
        "category": "Keyword",
        "query": "mouse 6-month warranty sensor click-switch failure",
        "expected_cards": {4},
    },
    {
        "id": "Q3",
        "category": "Keyword",
        "query": "Outstation delivery Anuradhapura Kandy Galle 1,200 LKR",
        "expected_cards": {6, 7},
    },
    {
        "id": "Q4",
        "category": "Keyword",
        "query": "free insured express shipping above 250,000 LKR",
        "expected_cards": {7, 8},
    },
    {
        "id": "Q4b",
        "category": "Keyword",
        "query": "motherboard",
        "expected_cards": {1, 2},
    },
    {
        "id": "Q5",
        "category": "Semantic",
        "query": "My screen has tiny black dots that won't light up",
        "expected_cards": {3},
    },
    {
        "id": "Q6",
        "category": "Semantic",
        "query": "Can I get my money back if the box is still factory sealed?",
        "expected_cards": {8, 9},
    },
    {
        "id": "Q7",
        "category": "Semantic",
        "query": "What happens if water or coffee gets spilled on my device?",
        "expected_cards": {4, 5},
    },
    {
        "id": "Q8",
        "category": "Semantic",
        "query": "How fast does a package arrive inside the capital city Colombo?",
        "expected_cards": {5, 6},
    },
    {
        "id": "Q9",
        "category": "Out-of-Scope",
        "query": "pepperoni pizza restaurant catering menu",
        "expected_cards": set(),
    },
    {
        "id": "Q10",
        "category": "Out-of-Scope",
        "query": "cheap airline flight tickets to Tokyo Japan",
        "expected_cards": set(),
    },
]


def evaluate_retriever(
    name: str,
    retriever_fn: Callable,
    chunks: list[dict],
    top_k: int = 2,
) -> dict:
    """Runs all 10 benchmark queries against a single retriever and measures Hit Rate % and CPU Latency (ms)."""
    hits = 0
    keyword_hits = 0
    semantic_hits = 0
    guard_hits = 0
    total_latency_ms = 0.0

    print(f"\n--- Benchmarking: {name} ---")
    for item in BENCHMARK_SUITE:
        query = item["query"]
        expected: set[int] = item["expected_cards"]
        category = item["category"]

        t0 = time.perf_counter()
        results = retriever_fn(query=query, chunks=chunks, top_k=top_k)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        total_latency_ms += latency_ms

        found_ids = [card["chunk_id"] for card in results]

        if len(expected) == 0:
            is_hit = len(found_ids) == 0
        else:
            is_hit = any(cid in expected for cid in found_ids)

        if is_hit:
            hits += 1
            if category == "Keyword":
                keyword_hits += 1
            elif category == "Semantic":
                semantic_hits += 1
            elif category == "Out-of-Scope":
                guard_hits += 1

        status = "PASS" if is_hit else "MISS"
        print(
            f"[{item['id']} | {category:12s}] {status:4s} | "
            f"Found: {str(found_ids):8s} | Expected: {str(sorted(expected)):8s} | "
            f"{latency_ms:6.2f} ms"
        )

    total_queries = len(BENCHMARK_SUITE)
    hit_rate = round((hits / total_queries) * 100.0, 1)
    avg_latency_ms = round(total_latency_ms / total_queries, 2)

    return {
        "retriever": name,
        "overall_hit_rate_pct": hit_rate,
        "keyword_hits": f"{keyword_hits}/5",
        "semantic_hits": f"{semantic_hits}/4",
        "out_of_scope_guards": f"{guard_hits}/2",
        "avg_cpu_latency_ms": avg_latency_ms,
    }


def run_all_benchmarks() -> None:
    """Loads policy chunks, warms up the ONNX model, benchmarks all 3 engines, and saves CSV."""
    chunks = load_and_chunk_file(file_path=POLICY_FILE, chunk_size=25, overlap=5)
    print(f"Loaded {len(chunks)} policy cards from '{POLICY_FILE.name}'.")

    print("Warming up local ONNX embedding model (BAAI/bge-small-en-v1.5)...")
    _ = retrieve_semantic_chunks(query="warmup query", chunks=chunks, top_k=1)

    retrievers = [
        ("1. Keyword-Only (Lexical)", retrieve_top_chunks),
        ("2. Neural-Only (FastEmbed ONNX)", retrieve_semantic_chunks),
        ("3. 50/50 Hybrid RAG (Boss)", retrieve_hybrid_chunks),
    ]

    summary_rows: list[dict] = []
    for name, func in retrievers:
        metrics = evaluate_retriever(name=name, retriever_fn=func, chunks=chunks, top_k=2)
        summary_rows.append(metrics)

    fieldnames = [
        "retriever",
        "overall_hit_rate_pct",
        "keyword_hits",
        "semantic_hits",
        "out_of_scope_guards",
        "avg_cpu_latency_ms",
    ]
    with CSV_OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n" + "=" * 88)
    print("FINAL DAY 6 RETRIEVAL BENCHMARK SUMMARY (11 QUERIES ON LOCAL INTEL CORE i5 CPU)")
    print("=" * 88)
    print(
        f"{'Retriever Engine':34s} | {'Hit Rate':9s} | {'Keyword':7s} | "
        f"{'Semantic':8s} | {'Guards':6s} | {'Avg Latency':11s}"
    )
    print("-" * 88)
    for row in summary_rows:
        print(
            f"{row['retriever']:34s} | {row['overall_hit_rate_pct']:7.1f}%  | "
            f"{row['keyword_hits']:7s} | {row['semantic_hits']:8s} | "
            f"{row['out_of_scope_guards']:6s} | {row['avg_cpu_latency_ms']:8.2f} ms"
        )
    print("=" * 88)
    print(f"Saved CSV benchmark artifact to: {CSV_OUTPUT_PATH}")

    hybrid_row = summary_rows[2]
    keyword_row = summary_rows[0]
    assert hybrid_row["overall_hit_rate_pct"] > keyword_row["overall_hit_rate_pct"], (
        "FAIL: Expected 50/50 Hybrid RAG to beat Keyword-Only Hit Rate!"
    )
    print("ARCHITECT VERIFICATION PASSED: 50/50 Hybrid Engine outperforms Keyword-Only baseline!")


if __name__ == "__main__":
    run_all_benchmarks()