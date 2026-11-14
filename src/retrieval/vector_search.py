import math
import os
import sys
from pathlib import Path

# 1. Lock cache to D: drive & silence Windows symlink warning BEFORE importing fastembed
CACHE_DIR = "D:/AI_Vault/huggingface_cache"
os.environ["HF_HOME"] = CACHE_DIR
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from fastembed import TextEmbedding

# 2. Ensure Python can import loader.py from src/ingestion
INGESTION_DIR = Path(__file__).resolve().parent.parent / "ingestion"
if str(INGESTION_DIR) not in sys.path:
    sys.path.insert(0, str(INGESTION_DIR))

from loader import load_and_chunk_file

# 3. Initialize the neural embedding model once in memory
EMBEDDER = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    cache_dir=CACHE_DIR,
)


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Computes exact Cosine Similarity (0.0 to 1.0) between two 384-d vectors."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def retrieve_semantic_chunks(
    query: str,
    chunks: list[dict],
    top_k: int = 2,
    min_score: float = 0.58,
) -> list[dict]:
    """
    Embeds the customer query and all policy chunks using BAAI/bge-small-en-v1.5,
    returning the top_k chunks whose meaning score passes min_score.
    """
    if not query or not query.strip():
        raise ValueError("Error: query cannot be empty.")
    if not chunks:
        raise ValueError("Error: chunks list cannot be empty.")
    if top_k <= 0:
        raise ValueError("Error: top_k must be greater than 0.")

    # Convert query and all chunk texts into 384-number meaning vectors in one batch
    all_texts = [query.strip()] + [card["text"] for card in chunks]
    all_vectors = [vec.tolist() for vec in EMBEDDER.embed(all_texts)]

    query_vec = all_vectors[0]
    chunk_vecs = all_vectors[1:]

    scored_cards = []
    for card, c_vec in zip(chunks, chunk_vecs):
        sim_score = cosine_similarity(query_vec, c_vec)
        if sim_score >= min_score:
            card_copy = dict(card)
            card_copy["semantic_score"] = round(sim_score, 4)
            scored_cards.append(card_copy)

    scored_cards.sort(key=lambda item: item["semantic_score"], reverse=True)
    return scored_cards[:top_k]


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    policy_file = PROJECT_ROOT / "data" / "shop_policies.txt"

    cards = load_and_chunk_file(file_path=policy_file, chunk_size=25, overlap=5)

    # Test 1: ZERO exact keyword overlap with "monitor" or "dead pixels"!
    query_1 = "My screen has tiny black dots that won't light up"
    print(f"=== TEST 1: NEURAL MEANING SEARCH ('{query_1}') ===")
    results_1 = retrieve_semantic_chunks(query=query_1, chunks=cards, top_k=2)
    for match in results_1:
        print(
            f"Card {match['chunk_id']} | Semantic Score: {match['semantic_score']} | Text: {match['text']}"
        )

    # Test 2: Unrelated query bouncer check
    query_2 = "pepperoni pizza restaurant menu"
    print(f"\n=== TEST 2: UNRELATED QUERY BOUNCER ('{query_2}') ===")
    results_2 = retrieve_semantic_chunks(query=query_2, chunks=cards, top_k=2)
    print(f"Matches found: {len(results_2)} (Expected: 0)")