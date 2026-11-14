import os
import sys
from pathlib import Path

# 1. Silence Windows Hugging Face symlink warning & lock cache to D: drive
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HOME"] = "D:/AI_Vault/huggingface_cache"

# 2. Connect src/retrieval and src/ingestion to sys.path
RETRIEVAL_DIR = Path(__file__).resolve().parent
INGESTION_DIR = RETRIEVAL_DIR.parent / "ingestion"

for folder in (RETRIEVAL_DIR, INGESTION_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from loader import load_and_chunk_file
from retriever import retrieve_top_chunks
from vector_search import retrieve_semantic_chunks


def retrieve_hybrid_chunks(
    query: str,
    chunks: list[dict],
    top_k: int = 2,
) -> list[dict]:
    """
    Combines Keyword Search (exact matches) and Neural Vector Search (meaning matches)
    using a 50/50 normalized score, deduplicated by chunk_id.
    """
    # Input Bouncers
    if not query or not query.strip():
        raise ValueError("Error: query cannot be empty.")
    if not chunks:
        raise ValueError("Error: chunks list cannot be empty.")
    if top_k <= 0:
        raise ValueError("Error: top_k must be greater than 0.")

    # Ask both detectives to score all candidate chunks
    kw_results = retrieve_top_chunks(query=query, chunks=chunks, top_k=len(chunks))
    sem_results = retrieve_semantic_chunks(
        query=query, chunks=chunks, top_k=len(chunks), min_score=0.58
    )

    # Deduplicate and merge using chunk_id as the unique key
    merged_by_id: dict[int, dict] = {}

    for card in kw_results:
        cid = card["chunk_id"]
        merged_by_id[cid] = {
            "chunk_id": cid,
            "source": card["source"],
            "text": card["text"],
            "keyword_score": card["score"],
            "semantic_score": 0.0,
        }

    for card in sem_results:
        cid = card["chunk_id"]
        if cid in merged_by_id:
            merged_by_id[cid]["semantic_score"] = card["semantic_score"]
        else:
            merged_by_id[cid] = {
                "chunk_id": cid,
                "source": card["source"],
                "text": card["text"],
                "keyword_score": 0,
                "semantic_score": card["semantic_score"],
            }

    # Compute the 50/50 Hybrid Score on a 0.0 to 1.0 scale
    final_cards = []
    for card in merged_by_id.values():
        kw_norm = min(card["keyword_score"] * 0.25, 1.0)
        sem_score = card["semantic_score"]
        hybrid_score = round((0.5 * kw_norm) + (0.5 * sem_score), 4)

        if hybrid_score >= 0.20:
            card["hybrid_score"] = hybrid_score
            card["score"] = hybrid_score
            final_cards.append(card)

    final_cards.sort(key=lambda item: item["hybrid_score"], reverse=True)
    return final_cards[:top_k]


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    policy_file = PROJECT_ROOT / "data" / "shop_policies.txt"

    cards = load_and_chunk_file(file_path=policy_file, chunk_size=25, overlap=5)

    # Test 1: Exact Keyword + Meaning Match (Both Detectives Team Up!)
    q1 = "Anuradhapura outstation delivery cost"
    print(f"=== TEST 1: HYBRID EXACT + MEANING ('{q1}') ===")
    for m in retrieve_hybrid_chunks(query=q1, chunks=cards, top_k=2):
        print(
            f"Card {m['chunk_id']} | Hybrid: {m['hybrid_score']} "
            f"(KW: {m['keyword_score']}, Sem: {m['semantic_score']}) | Text: {m['text']}"
        )

    # Test 2: Pure Meaning Match (Zero Exact Keywords -> Neural Detective Saves It!)
    q2 = "My screen has tiny black dots that won't light up"
    print(f"\n=== TEST 2: HYBRID MEANING RESCUE ('{q2}') ===")
    for m in retrieve_hybrid_chunks(query=q2, chunks=cards, top_k=2):
        print(
            f"Card {m['chunk_id']} | Hybrid: {m['hybrid_score']} "
            f"(KW: {m['keyword_score']}, Sem: {m['semantic_score']}) | Text: {m['text']}"
        )

    # Test 3: Fake Policy Bouncer (Both Detectives Score 0)
    q3 = "pepperoni pizza restaurant menu"
    print(f"\n=== TEST 3: HYBRID BOUNCER ('{q3}') ===")
    res3 = retrieve_hybrid_chunks(query=q3, chunks=cards, top_k=2)
    print(f"Matches found: {len(res3)} (Expected: 0)")