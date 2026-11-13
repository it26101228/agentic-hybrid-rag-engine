import os
import sys
from pathlib import Path
from smolagents import OpenAIServerModel, ToolCallingAgent
from smolagents.models import ChatMessageToolCall, ChatMessageToolCallFunction

# 1. Ensure Python always finds tools.py inside src/agent
AGENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = AGENT_DIR.parent.parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from tools import (
    get_item_price,
    calculate_final_price,
    check_approval_status,
    search_shop_policies,
)


def load_env_file() -> None:
    """Loads environment variables from the root .env file into os.environ."""
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists() and env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            clean_line = line.strip()
            if clean_line and not clean_line.startswith("#") and "=" in clean_line:
                key, value = clean_line.split("=", 1)
                os.environ[key.strip()] = value.strip().strip('"').strip("'")


class GroqSafeModel(OpenAIServerModel):
    """
    Wrapper around OpenAIServerModel that:
    1. Forces tool_choice='auto' whenever tools are provided so Groq never throws 400 errors.
    2. Wraps plain-text final responses into a 'final_answer' tool call for ToolCallingAgent.
    """

    def generate(self, messages, **kwargs):
        if kwargs.get("tools_to_call_from"):
            kwargs["tool_choice"] = "auto"

        msg = super().generate(messages, **kwargs)

        if not msg.tool_calls and msg.content:
            msg.tool_calls = [
                ChatMessageToolCall(
                    id="call_final_answer",
                    type="function",
                    function=ChatMessageToolCallFunction(
                        name="final_answer",
                        arguments={"answer": msg.content},
                    ),
                )
            ]
        return msg


def build_shop_agent() -> ToolCallingAgent:
    """Builds and returns the 4-tool Agentic RAG shop assistant."""
    load_env_file()
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Error: GROQ_API_KEY is missing! Check your .env file.")

    model = GroqSafeModel(
        model_id="openai/gpt-oss-20b",
        api_base="https://api.groq.com/openai/v1",
        api_key=api_key,
    )

    agent = ToolCallingAgent(
        tools=[
            get_item_price,
            calculate_final_price,
            check_approval_status,
            search_shop_policies,
        ],
        model=model,
        max_steps=8,
    )
    return agent


if __name__ == "__main__":
    shop_agent = build_shop_agent()

    mission_prompt = (
        "You are the official Tech Vault Sri Lanka assistant. "
        "STRICT RULE: Only use facts returned by your tools. Never guess prices or policies. "
        "Customer Request: I want to buy a monitor. "
        "1. What is its final price in LKR including 18% VAT, and is it auto-approved? "
        "2. According to store policy, how many dead pixels on a monitor qualify for a free replacement? "
        "3. How much does outstation delivery to Anuradhapura cost and how many days does it take?"
    )

    print("=== RUNNING 4-TOOL AGENTIC RAG MISSION ===")
    final_response = shop_agent.run(mission_prompt)
    print("\n=== FINAL AGENT RESPONSE ===")
    print(final_response)