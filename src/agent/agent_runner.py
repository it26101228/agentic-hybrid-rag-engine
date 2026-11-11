import json
import os
import re
from smolagents import ToolCallingAgent, InferenceClientModel
from tools import get_item_price, calculate_final_price, check_approval_status


class GroqSafeModel(InferenceClientModel):
    def _prepare_completion_kwargs(self, *args, **kwargs):
        kw = super()._prepare_completion_kwargs(*args, **kwargs)
        if "tool_choice" in kw:
            kw["tool_choice"] = "auto"
        return kw


model = GroqSafeModel(
    model_id="openai/gpt-oss-20b",
    provider="groq",
    token=os.environ["GROQ_API_KEY"],
)

agent = ToolCallingAgent(
    tools=[get_item_price, calculate_final_price, check_approval_status],
    model=model,
    max_steps=3,
    verbosity_level=2,
)

task = (
    "I want to buy a laptop. Please find its base price, calculate the final "
    "price with VAT, and check if it requires manager approval."
)

print("--- STARTING AGENT EXECUTION ---")
try:
    result = agent.run(task)
    print("\n--- AGENT FINAL ANSWER ---")
    print(result)
except Exception as e:
    match = re.search(r'"answer"\s*:\s*"([^"]+)"', str(e))
    if match:
        print("\n--- AGENT FINAL ANSWER ---")
        print(match.group(1).encode("utf-8").decode("unicode_escape"))
    else:
        raise e