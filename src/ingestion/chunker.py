def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Slices a long document string into smaller overlapping word chunks for RAG ingestion.

    Args:
        text: The raw document text to slice.
        chunk_size: The maximum number of words per chunk.
        overlap: The number of words shared between consecutive chunks.
    """
    # --- BOUNCER 1: Block empty or whitespace-only text ---
    if not text.strip():
        raise ValueError("Error: text cannot be empty.")

    # --- BOUNCER 2: Block zero/negative chunk_size or negative overlap ---
    if chunk_size <= 0 or overlap < 0:
        raise ValueError("Error: chunk_size must be > 0 and overlap must be >= 0.")

    # --- BOUNCER 3: Block the Infinite Loop Trap (overlap must be smaller than chunk_size) ---
    if overlap >= chunk_size:
        raise ValueError("Error: overlap must be strictly smaller than chunk_size.")

    # --- THE KITCHEN: Slice the words with overlap ---
    words = text.split()
    step_size = chunk_size - overlap
    chunks = []

    for i in range(0, len(words), step_size):
        card_words = words[i : i + chunk_size]
        card_text = " ".join(card_words)
        chunks.append(card_text)

        # Stop Sign: If this card already reached the very last word, stop slicing!
        if i + chunk_size >= len(words):
            break

    return chunks


if __name__ == "__main__":
    # --- TEST 1: The 10-Word Paper Blueprint Test ---
    sample_policy = "Our shop gives a three year warranty on all laptops"
    result_chunks = chunk_text(text=sample_policy, chunk_size=4, overlap=1)

    print("=== TEST 1: VALID CHUNKING (chunk_size=4, overlap=1) ===")
    for index, card in enumerate(result_chunks, start=1):
        print(f"Card {index}: '{card}'")

    # --- TEST 2: Prove Bouncer 3 Blocks the Infinite Loop Trap ---
    print("\n=== TEST 2: BOUNCER 3 STRESS TEST (chunk_size=4, overlap=4) ===")
    try:
        chunk_text(text=sample_policy, chunk_size=4, overlap=4)
    except ValueError as err:
        print(f"Bouncer successfully blocked bad input -> {err}")