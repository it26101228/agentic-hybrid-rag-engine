import sys
from pathlib import Path

# 1. Ensure Python always finds chunker.py inside src/ingestion
INGESTION_DIR = Path(__file__).resolve().parent
if str(INGESTION_DIR) not in sys.path:
    sys.path.insert(0, str(INGESTION_DIR))

from chunker import chunk_text


def load_and_chunk_file(
    file_path: Path, chunk_size: int = 25, overlap: int = 5
) -> list[dict]:
    """
    Loads a text document from disk, slices it into overlapping word chunks,
    and packages each chunk into a metadata dictionary for RAG retrieval.

    Args:
        file_path: Path object pointing to the .txt document.
        chunk_size: Maximum number of words per chunk (default: 25).
        overlap: Number of overlapping words between consecutive chunks (default: 5).
    """
    # --- BOUNCER 1: Location Check (Does the path exist on the hard drive?) ---
    if not file_path.exists():
        raise FileNotFoundError(f"Error: File not found at '{file_path}'.")

    # --- BOUNCER 2: Type Check (Is it a real file and NOT a folder?) ---
    if not file_path.is_file():
        raise ValueError(f"Error: Expected a file, but '{file_path}' is a folder.")

    # --- THE KITCHEN: Read file -> Slice words -> Package Dictionary Cards ---
    raw_text = file_path.read_text(encoding="utf-8")
    raw_chunks = chunk_text(text=raw_text, chunk_size=chunk_size, overlap=overlap)

    chunk_records = []
    for index, card_text in enumerate(raw_chunks, start=1):
        chunk_card = {
            "chunk_id": index,
            "source": file_path.name,
            "text": card_text,
        }
        chunk_records.append(chunk_card)

    return chunk_records


if __name__ == "__main__":
    # Locate the project root folder (D:\AI_Vault\agentic-hybrid-rag-engine)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    policy_file = PROJECT_ROOT / "data" / "shop_policies.txt"

    print("=== TEST 1: LOAD & CHUNK SHOP POLICIES MANUAL ===")
    cards = load_and_chunk_file(file_path=policy_file, chunk_size=25, overlap=5)
    print(f"Total dictionary cards created: {len(cards)}\n")

    # Print the first 3 dictionary cards to inspect metadata and overlap
    for card in cards[:3]:
        print(card)

    print("\n=== TEST 2: FOLDER BOUNCER STRESS TEST (Passing 'data' folder instead of a file) ===")
    data_folder = PROJECT_ROOT / "data"
    try:
        load_and_chunk_file(file_path=data_folder)
    except ValueError as err:
        print(f"Bouncer successfully blocked folder input -> {err}")