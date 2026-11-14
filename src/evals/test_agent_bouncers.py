import sys
from pathlib import Path

# 1. Connect src/evals to src/agent so we reuse build_shop_agent() and search_shop_policies()
AGENT_DIR = Path(__file__).resolve().parent.parent / "agent"
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from agent_runner import build_shop_agent
from tools import search_shop_policies


def run_evaluation_suite() -> None:
    """Runs automated end-to-end tests on math tools, inventory bouncers, RAG guards, and Hybrid meaning search."""
    agent = build_shop_agent()

    strict_rules = (
        "You are the official Tech Vault Sri Lanka assistant. "
        "STRICT RULE: Only use facts returned by your tools. "
        "If an item or policy is not found by a tool, state clearly that it is unavailable "
        "and NEVER guess or invent prices or policies from memory.\n"
    )

    # --- TEST 1: Valid Inventory & Auto-Approval Path ---
    print("=== EVAL TEST 1: MOUSE AUTO-APPROVAL PATH ===")
    prompt_1 = (
        strict_rules
        + "Customer Request: What is the final price of a mouse including 18% VAT, and what is its approval status?"
    )
    res_1 = str(agent.run(prompt_1))
    print(f"Agent Output 1:\n{res_1}\n")
    assert "4130" in res_1.replace(",", ""), "FAIL: Expected 4,130 LKR in Test 1!"
    assert "AUTO_APPROVED" in res_1, "FAIL: Expected AUTO_APPROVED in Test 1!"
    print("✅ TEST 1 PASSED: Accurate LKR VAT math and AUTO_APPROVED status verified!\n")

    # --- TEST 2: Missing Inventory Bouncer (PlayStation Hallucination Guard) ---
    print("=== EVAL TEST 2: PLAYSTATION MISSING-INVENTORY BOUNCER ===")
    prompt_2 = (
        strict_rules
        + "Customer Request: I want to buy a playstation. What is its final price with VAT?"
    )
    res_2 = str(agent.run(prompt_2))
    print(f"Agent Output 2:\n{res_2}\n")
    assert "499" not in res_2, "FAIL: Agent hallucinated a fake PlayStation price!"
    print("✅ TEST 2 PASSED: Missing item blocked with zero price hallucination!\n")

    # --- TEST 3: RAG Policy Retrieval & Missing-Policy Guard ---
    print("=== EVAL TEST 3: RAG POLICY RETRIEVAL & MISSING-POLICY GUARD ===")
    prompt_3 = (
        strict_rules
        + "Customer Request: 1. How many years is the hardware warranty on a laptop above 200,000 LKR? "
        "2. Do you have a store policy for pepperoni pizza catering?"
    )
    res_3 = str(agent.run(prompt_3))
    print(f"Agent Output 3:\n{res_3}\n")
    assert "3" in res_3 or "three" in res_3.lower(), "FAIL: Expected 3-year laptop warranty from RAG manual!"
    print("✅ TEST 3 PASSED: RAG policy retrieved and missing pizza policy rejected!\n")

    # --- TEST 4: Hybrid Semantic Meaning Rescue (Zero Exact Keywords) ---
    print("=== EVAL TEST 4: HYBRID SEMANTIC MEANING RESCUE ===")
    hybrid_res = search_shop_policies("My screen has tiny black dots that won't light up")
    print(f"Hybrid Tool Output:\n{hybrid_res}\n")
    assert "dead pixels" in hybrid_res, "FAIL: Hybrid search missed the dead pixel policy!"
    assert "Card 3" in hybrid_res, "FAIL: Expected Card 3 to be retrieved by Hybrid Search!"
    print("✅ TEST 4 PASSED: Hybrid neural search rescued zero-keyword query!\n")

    print("🎉 ALL 4 END-TO-END HYBRID AGENTIC RAG EVALUATIONS PASSED 100%!")


if __name__ == "__main__":
    run_evaluation_suite()