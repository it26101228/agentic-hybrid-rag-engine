import sys
from pathlib import Path
from smolagents import tool

# 1. Connect src/agent/tools.py to src/ingestion and src/retrieval
SRC_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SRC_DIR.parent
INGESTION_DIR = SRC_DIR / "ingestion"
RETRIEVAL_DIR = SRC_DIR / "retrieval"
POLICY_FILE = PROJECT_ROOT / "data" / "shop_policies.txt"

for folder in (INGESTION_DIR, RETRIEVAL_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from loader import load_and_chunk_file
from hybrid_retriever import retrieve_hybrid_chunks


@tool
def get_item_price(item_name: str) -> str:
    """
    Looks up the base price of an electronic item in the shop inventory in LKR.

    Args:
        item_name: The name of the item to look up (e.g., 'laptop', 'monitor', 'mouse').
    """
    prices = {
        "laptop": 250000.0,
        "monitor": 75000.0,
        "mouse": 3500.0,
    }
    clean_name = item_name.strip().lower()
    if clean_name not in prices:
        return f"Error: '{item_name}' is not in the shop inventory. Do NOT guess or invent a price."
    return str(prices[clean_name])


@tool
def calculate_final_price(original_price: float) -> float:
    """
    Calculates the final price in LKR after adding 18% Sri Lankan VAT.

    Args:
        original_price: The base price of the item before tax.
    """
    if original_price <= 0:
        raise ValueError("Error: Price must be greater than 0.")
    return round(original_price * 1.18, 2)


@tool
def check_approval_status(final_price: float) -> str:
    """
    Checks whether a purchase requires manager approval based on the final VAT-inclusive price.

    Args:
        final_price: The final price in LKR including 18% VAT.
    """
    if final_price <= 0:
        raise ValueError("Error: Final price must be greater than 0.")
    if final_price > 100000:
        return "MANAGER_APPROVAL_REQUIRED"
    return "AUTO_APPROVED"


@tool
def search_shop_policies(query: str) -> str:
    """
    Searches the official store policy and warranty manual using Hybrid RAG
    (combining exact keyword search and neural semantic meaning search) for rules on
    warranties, dead pixels, screen defects, returns, refunds, and delivery across Sri Lanka.

    Args:
        query: Search query or customer problem description (e.g., 'Anuradhapura delivery cost' or 'screen has black dots').
    """
    if not query or not query.strip():
        raise ValueError("Error: search query cannot be empty.")

    chunks = load_and_chunk_file(file_path=POLICY_FILE, chunk_size=25, overlap=5)
    results = retrieve_hybrid_chunks(query=query, chunks=chunks, top_k=2)

    if not results:
        return "No matching policies or instructions found. Do NOT guess or invent a policy or instructions."

    formatted_cards = []
    for match in results:
        formatted_cards.append(
            f"[Card {match['chunk_id']} | Source: {match['source']} | Hybrid Score: {match['hybrid_score']}]: {match['text']}"
        )

    return "\n".join(formatted_cards)


if __name__ == "__main__":
    print("=== TEST 1: HYBRID KEYWORD + MEANING (Laptop Warranty) ===")
    print(search_shop_policies("laptop warranty years motherboard"))

    print("\n=== TEST 2: HYBRID MEANING RESCUE (Zero Exact Keywords!) ===")
    print(search_shop_policies("My screen has tiny black dots that won't light up"))

    print("\n=== TEST 3: MISSING POLICY GUARD (Pepperoni Pizza) ===")
    print(search_shop_policies("pepperoni pizza restaurant"))