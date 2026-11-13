import re
import sys
from pathlib import Path

# 1. Allow importing loader.py from src/ingestion
INGESTION_DIR = Path(__file__).resolve().parent.parent / "ingestion"
if str(INGESTION_DIR) not in sys.path:
    sys.path.insert(0, str(INGESTION_DIR))

from loader import load_and_chunk_file

# Common filler words (stopwords) that should not earn keyword points
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does",
    "for", "from", "how", "i", "if", "in", "is", "it", "much", "my",
    "of", "on", "or", "the", "to", "what", "when", "where", "which", "with",
}


def extract_keywords(text: str) -> set[str]:
    """
    Cleans text into lowercase words, removes punctuation and filler stopwords,
    and normalizes simple plurals (e.g., 'costs' -> 'cost', 'laptops' -> 'laptop').
    """
    raw_words = re.findall(r"[a-z0-9]+", text.lower())
    keywords = set()

    for word in raw_words:
        if word in STOPWORDS:
            continue
        # Normalize simple plural 's' so 'costs' matches 'cost' and 'laptops' matches 'laptop'
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        keywords.add(word)

    return keywords


def retrieve_top_chunks(
    query: str, chunks: list[dict], top_k: int = 2
) -> list[dict]:
    """
    Scores each dictionary chunk against the customer's query keywords
    and returns the top_k highest-scoring chunks (excluding score 0 cards).

    Args:
        query: The customer's question.
        chunks: List of metadata chunk dictionaries from load_and_chunk_file().
        top_k: Maximum number of matching cards to return.
    """
    # --- BOUNCER 1: Block empty query ---
    if not query.strip():
        raise ValueError("Error: query cannot be empty.")

    # --- BOUNCER 2: Block empty chunk list or invalid top_k ---
    if not chunks:
        raise ValueError("Error: chunks list cannot be empty.")
    if top_k <= 0:
        raise ValueError("Error: top_k must be greater than 0.")

    # --- THE KITCHEN: Score each card by unique keyword overlap ---
    query_keywords = extract_keywords(query)
    scored_chunks = []

    for card in chunks:
        card_keywords = extract_keywords(card["text"])
        matched_words = query_keywords.intersection(card_keywords)
        score = len(matched_words)

        # Only keep cards that matched at least 1 meaningful keyword
        if score > 0:
            scored_card = {
                "chunk_id": card["chunk_id"],
                "source": card["source"],
                "score": score,
                "matched_keywords": sorted(list(matched_words)),
                "text": card["text"],
            }
            scored_chunks.append(scored_card)

    # Sort cards from highest score to lowest score
    scored_chunks.sort(key=lambda item: item["score"], reverse=True)

    return scored_chunks[:top_k]


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    policy_file = PROJECT_ROOT / "data" / "shop_policies.txt"

    # Load all 9 cards from the shop manual
    all_cards = load_and_chunk_file(file_path=policy_file, chunk_size=25, overlap=5)

    # Test Query 1: Your Anuradhapura Delivery Question
    question_1 = "What is the courier delivery cost to Anuradhapura?"
    print(f"=== QUERY 1: '{question_1}' ===")
    top_matches_1 = retrieve_top_chunks(query=question_1, chunks=all_cards, top_k=2)
    for rank, match in enumerate(top_matches_1, start=1):
        print(
            f"Rank #{rank} | Card {match['chunk_id']} | Score: {match['score']} "
            f"| Matched: {match['matched_keywords']}\n   Text: '{match['text']}'"
        )

    # Test Query 2: Dead Pixel Monitor Question
    question_2 = "How many dead pixels on a monitor get a free replacement?"
    print(f"\n=== QUERY 2: '{question_2}' ===")
    top_matches_2 = retrieve_top_chunks(query=question_2, chunks=all_cards, top_k=2)
    for rank, match in enumerate(top_matches_2, start=1):
        print(
            f"Rank #{rank} | Card {match['chunk_id']} | Score: {match['score']} "
            f"| Matched: {match['matched_keywords']}\n   Text: '{match['text']}'"
        )