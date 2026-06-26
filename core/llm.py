"""
Local LLM client for llama.cpp.

This project targets **local** inference only: a model downloaded from Hugging Face
(GGUF) and served by ``llama-server`` (the server that ships with llama.cpp).

Start the server, e.g.:

    llama-server -m ./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --port 8080
"""

from __future__ import annotations

import os

from langchain_openai import ChatOpenAI


def build_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Return a chat client wired to the local llama.cpp server."""
    base_url = os.getenv("LLAMA_CPP_BASE_URL", "http://localhost:8080/v1")
    # llama-server ignores the model field and serves whatever GGUF was loaded; we
    # pass the configured name purely for clearer logs/traces.
    model_name = os.getenv("LLAMA_MODEL", "local-gguf")
    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=os.getenv("LLAMA_CPP_API_KEY", "not-needed"),  # local server: auth disabled
        temperature=temperature,
    )
