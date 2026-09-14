"""
Thin, backend-agnostic LLM client. Works against any OpenAI-compatible
chat completions endpoint -- Ollama (local), Bionic (e.g. on a separate
Windows PC over the LAN), LM Studio, vLLM, etc. Only LLM_BASE_URL /
LLM_MODEL need to change; no code changes when switching backends.

This keeps the LLM genuinely at the edges of the pipeline (per the
project's locked architecture): intent_parser.py and explainer.py both
import get_client() from here rather than each owning their own
connection logic.
"""
from __future__ import annotations

import os

from openai import OpenAI


def get_client() -> tuple[OpenAI, str]:
    """
    Returns (client, model_name). Reads config from environment variables
    (see .env.example) so switching backends -- e.g. LM Studio on the
    Windows PC over the LAN vs. local Ollama -- is a config change, not a
    code change. LM Studio exposes an OpenAI-compatible API at /v1 on
    port 1234 by default (GET /v1/models, POST /v1/chat/completions),
    no API key required.
    """
    base_url = os.environ.get("LLM_BASE_URL", "http://localhost:1234/v1")
    model = os.environ.get("LLM_MODEL", "local-model")
    api_key = os.environ.get("LLM_API_KEY", "lm-studio")

    client = OpenAI(base_url=base_url, api_key=api_key)
    return client, model


if __name__ == "__main__":
    # Quick connectivity check -- confirms whichever backend is configured
    # actually responds, without touching any project-specific logic.
    client, model = get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=10,
    )
    print(f"Backend: {os.environ.get('LLM_BASE_URL', '(default)')}")
    print(f"Model: {model}")
    print(f"Response: {response.choices[0].message.content}")
