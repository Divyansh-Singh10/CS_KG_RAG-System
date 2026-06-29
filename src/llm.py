import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

# ── config ────────────────────────────────────────────────────────────────────

GEMINI_MODEL       = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_API_KEY     = os.getenv("GEMINI_API_KEY")
TEMPERATURE        = 0.2    # low = more factual, less creative (good for RAG)
MAX_OUTPUT_TOKENS  = 512    # enough for a detailed answer, not wasteful


# ── LLM class ─────────────────────────────────────────────────────────────────

class LLM:

    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY not found — check your .env file"
            )

        self.client = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GEMINI_API_KEY,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
        print(f"LLM ready — model: {GEMINI_MODEL}")

    def generate(self, system_prompt: str, user_message: str) -> str:
        """
        generates a response given a system prompt and user message
        returns the response as a plain string
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        response = self.client.invoke(messages)
        return response.content.strip()

    def generate_with_history(self,
                               system_prompt: str,
                               history: list[dict],
                               user_message: str) -> str:
        """
        generates a response with conversation history
        history format: [{"role": "user"|"assistant", "content": "..."}]
        """
        from langchain_core.messages import AIMessage

        messages = [SystemMessage(content=system_prompt)]

        for turn in history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))

        messages.append(HumanMessage(content=user_message))

        response = self.client.invoke(messages)
        return response.content.strip()


# ── run directly ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    llm = LLM()

    # test 1 — basic generation
    print("\n--- test 1: basic generation ---")
    response = llm.generate(
        system_prompt="You are a helpful assistant. Answer concisely.",
        user_message="What is a knowledge graph?"
    )
    print(response)

    # test 2 — confirm temperature is working (factual, not creative)
    print("\n--- test 2: factual response ---")
    response = llm.generate(
        system_prompt="You are a factual assistant. Only state confirmed facts.",
        user_message="What color is the sky?"
    )
    print(response)