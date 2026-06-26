"""
Local LLM client for llama.cpp.

This project targets **local** inference only: a model downloaded from Hugging Face
(GGUF) and served by ``llama-server`` (the server that ships with llama.cpp).

Why we still import ``langchain_openai.ChatOpenAI``
---------------------------------------------------
``llama-server`` exposes an **OpenAI-compatible** HTTP API at ``/v1`` (this is a
llama.cpp feature, not a dependency on OpenAI). ``ChatOpenAI`` is simply the most
convenient, well-supported client for *any* OpenAI-compatible endpoint. Pointed at
``http://localhost:8080/v1`` with a throwaway API key, no request ever leaves the
machine and OpenAI's servers are never contacted. We keep the client but make the
intent explicit here, in one place.

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
