import os
import re
import sys
from pathlib import Path
from smolagents import ToolCallingAgent, InferenceClientModel

# 1. Tell Python how to find the 'src/agent' folder from inside 'src/evals'
AGENT_DIR = Path(__file__).resolve().parent.parent / "agent"
sys.path.insert(0, str(AGENT_DIR))

from tools import get_item_price, calculate_final_price, check_approval_status


# 2. Safely load GROQ_API_KEY from the .env file in the project root
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
if env_path.is_file():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        clean_line = line.strip()
        if clean_line.startswith("GROQ_API_KEY="):
            os.environ["GROQ_API_KEY"] = clean_line.split("=", 1)[1].strip().strip('"').strip("'")


class GroqSafeModel(InferenceClientModel):
    def _prepare_completion_kwargs(self, *args, **kwargs):
        kw = super()._prepare_completion_kwargs(*args, **kwargs)
        if "tool_choice" in kw:
            kw["tool_choice"] = "auto"
        return kw


def run_eval_case(test_name: str, prompt: str):
    print("\n==========================================")
    print(f"RUNNING EVAL: {test_name}")
    print(f"PROMPT: {prompt}")
    print("==========================================")

    model = GroqSafeModel(
        model_id="openai/gpt-oss-20b",
        provider="groq",
        token=os.environ["GROQ_API_KEY"],
    )

    agent = ToolCallingAgent(
        tools=[get_item_price, calculate_final_price, check_approval_status],
        model=model,
        max_steps=3,
        verbosity_level=1,
    )

    strict_prompt = (
        f"{prompt}\n"
        "STRICT RULES: Always report currency in LKR (18% VAT). "
        "Never invent or guess prices from your own knowledge. "
        "If get_item_price returns an inventory error, do NOT retry other names—"
        "immediately reply that the item is out of stock."
    )

    try:
        result = agent.run(strict_prompt)
        print(f"\n[EVAL RESULT - {test_name}]:\n{result}")
    except Exception as e:
        match = re.search(r'"answer"\s*:\s*"([^"]+)"', str(e))
        if match:
            clean_ans = match.group(1).encode("utf-8").decode("unicode_escape")
            print(f"\n[EVAL RESULT - {test_name}]:\n{clean_ans}")
        else:
            print(f"\n[EVAL CAUGHT EXCEPTION - {test_name}]:\n{e}")


if __name__ == "__main__":
    # Eval 1: Low-cost item -> Should return 3,500 LKR base, 4,130 LKR with VAT, and AUTO_APPROVED
    run_eval_case(
        "Test 1: Auto-Approved Cheap Item (Mouse)",
        "I want to buy a mouse. Find its base price, calculate the final price with VAT, and check if it requires manager approval.",
    )

    # Eval 2: Missing item -> Should trigger your ValueError bouncer and refuse to make up a price
    run_eval_case(
        "Test 2: Missing Inventory Bouncer (PlayStation)",
        "I want to buy a playstation. Find its base price, calculate the final price with VAT, and check if it requires manager approval.",
    )