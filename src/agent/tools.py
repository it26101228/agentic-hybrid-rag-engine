from smolagents import tool


@tool
def get_item_price(item_name: str) -> float:
    """
    Finds the base price of an item in LKR from the shop inventory.

    Args:
        item_name: The name of the item before getting the relevant price.
    """
    prices = {
        "laptop": 250000.0,
        "monitor": 45000.0,
        "mouse": 3500.0,
    }

    price = prices.get(item_name.lower())

    if price is None:
        raise ValueError(f"Error: '{item_name}' is not in the shop inventory.")

    return round(price, 2)


@tool
def calculate_final_price(original_price: float) -> float:
    """
    Calculate the final price in LKR after adding 18% VAT. Use this tool whenever someone asks for the final price of an item.

    Args:
        original_price: The starting price of the item in LKR before adding VAT.
    """
    if original_price <= 0:
        raise ValueError("Error: original_price must be greater than 0 LKR.")

    final_price = original_price + original_price * 0.18
    return round(final_price, 2)


@tool
def check_approval_status(final_price: float) -> str:
    """
    Checks if a final price in LKR requires manager approval (strictly above 200,000 LKR) or is auto-approved.

    Args:
        final_price: The final amount of LKR to be paid by the customer.
    """
    if final_price > 200000.0:
        return "MANAGER_APPROVAL_REQUIRED"
    else:
        return "AUTO_APPROVED"


if __name__ == "__main__":
    # --- QUICK LOCAL TEST (Only runs when tools.py is executed directly) ---
    item = "Laptop"
    base = get_item_price(item)
    with_vat = calculate_final_price(base)
    status = check_approval_status(with_vat)
    print(f"Pipeline Result for {item}: Base={base} LKR -> With VAT={with_vat} LKR -> Status={status}")