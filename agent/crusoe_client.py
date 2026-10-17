"""Shared Crusoe client. Every agent calls the model through chat() or chat_json()."""
import json
import os
import random
import re
import time

from dotenv import load_dotenv
from openai import InternalServerError, OpenAI, RateLimitError

load_dotenv()

client = OpenAI(
    base_url=os.getenv("CRUSOE_BASE_URL", "https://api.inference.crusoecloud.com/v1"),
    api_key=os.environ["CRUSOE_API_KEY"],
)

PLANNER_MODEL = os.getenv("PLANNER_MODEL", "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B")
WORKER_MODEL = os.getenv("WORKER_MODEL", "nvidia/Nemotron-3.5-Lightning-30B-A3B")


def chat(messages, model=WORKER_MODEL, max_tokens=4096, max_tries=6, **kwargs) -> str:
    """Call the model and return its text. Retries on 429/503 with jittered backoff.

    Nemotron models think before answering, and thinking uses up max_tokens.
    Keep max_tokens generous (4096+) or the answer can come back empty.
    """
    for attempt in range(max_tries):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=1.0, top_p=0.95, **kwargs
            )
            choice = resp.choices[0]
            if not choice.message.content and choice.finish_reason == "length":
                raise RuntimeError(f"{model} ran out of max_tokens while thinking; raise max_tokens")
            return choice.message.content or ""
        except (RateLimitError, InternalServerError):
            if attempt == max_tries - 1:
                raise
            time.sleep(min(60, 2 ** attempt) + random.uniform(0, 0.5))


def chat_json(messages, model=WORKER_MODEL, **kwargs):
    """Like chat(), but parses a JSON object out of the reply (tolerates ```json fences)."""
    text = chat(messages, model=model, **kwargs)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in model reply:\n{text[:500]}")
    return json.loads(match.group(0))
