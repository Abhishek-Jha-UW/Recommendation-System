"""Optional OpenAI explanations — only called when user clicks and key is present."""

from __future__ import annotations

import json
from typing import Any


def explain_run_json(evidence: dict[str, Any], api_key: str, model: str) -> str:
    """
    One short grounded explanation. Caller must pass only computed facts in evidence.
    """
    if not api_key or not str(api_key).strip():
        raise ValueError("OpenAI API key is missing or empty.")
    if not model or not str(model).strip():
        raise ValueError("OpenAI model name is missing or empty.")

    from openai import OpenAI

    client = OpenAI(api_key=str(api_key).strip())
    payload = json.dumps(evidence, default=str)[:12000]
    resp = client.chat.completions.create(
        model=model,
        temperature=0.3,
        max_tokens=280,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a data science assistant. Use ONLY the JSON facts provided. "
                    "Do not invent users, items, or numbers. If facts are insufficient, say so in one sentence. "
                    "Respond in 2-4 short bullet points."
                ),
            },
            {
                "role": "user",
                "content": "Facts (JSON):\n" + payload,
            },
        ],
    )
    choice = resp.choices[0].message
    text = (choice.content or "").strip()
    return text or "(Empty model response.)"
