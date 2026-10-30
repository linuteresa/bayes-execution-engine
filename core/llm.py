"""
Local LLM client for llama.cpp.

This project targets **local** inference only: a model downloaded from Hugging Face
(GGUF) and served by ``llama-server`` (the server that ships with llama.cpp).

Start the server, e.g.:

    llama-server -m ./models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --port 8080
"""

from __future__ import annotations

import math
import os

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from core.runtime import positive_float, positive_int


def build_llm(temperature: float = 0.0, *, logprobs: bool = False,
              timeout: float | None = None) -> ChatOpenAI:
    """Return a chat client wired to the local llama.cpp server.

    Set ``logprobs=True`` to ask the server for per-token logprobs. The calibration
    eval uses them for its mean-token-logprob baseline (``eval/answerers.py``); the
    agent path leaves them off since nothing downstream reads them. Not every
    llama-server build honours the flag, so callers must tolerate their absence.
    """
    base_url = os.getenv("LLAMA_CPP_BASE_URL", "http://localhost:8080/v1")
    # llama-server ignores the model field and serves whatever GGUF was loaded; we
    # pass the configured name purely for clearer logs/traces.
    model_name = os.getenv("LLAMA_MODEL", "local-gguf")
    if not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")
    request_timeout = positive_float("LLM_TIMEOUT_SECONDS", 60)
    if timeout is not None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise TimeoutError("Execution deadline exceeded")
        request_timeout = min(request_timeout, timeout)
    kwargs = {}
    if logprobs:
        kwargs["logprobs"] = True
    return ChatOpenAI(
        model=model_name,
        base_url=base_url,
        api_key=os.getenv("LLAMA_CPP_API_KEY", "not-needed"),  # local server: auth disabled
        temperature=temperature,
        timeout=request_timeout,
        max_tokens=positive_int("LLM_MAX_TOKENS", 2048, maximum=32768),
        max_retries=0,  # job retries own the budget; hidden SDK retries cannot extend it
        **kwargs,
    )


def build_embeddings(*, timeout: float | None = None) -> OpenAIEmbeddings | None:
    """Embedding client for meaning-based sample agreement, or ``None`` to use word overlap.

    Set ``EMBEDDINGS_BASE_URL`` to a second llama-server started with ``--embeddings``:

        llama-server -hf second-state/All-MiniLM-L6-v2-Embedding-GGUF:Q8_0 --embeddings --port 8081
    """
    base_url = os.getenv("EMBEDDINGS_BASE_URL")
    if not base_url:
        return None
    request_timeout = positive_float("LLM_TIMEOUT_SECONDS", 60)
    if timeout is not None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise TimeoutError("Execution deadline exceeded")
        request_timeout = min(timeout, request_timeout)
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDINGS_MODEL", "local-embeddings"),
        base_url=base_url,
        api_key=os.getenv("LLAMA_CPP_API_KEY", "not-needed"),
        # The default pre-tokenizes with tiktoken (OpenAI vocab, network download);
        # llama-server tokenizes with the model's own vocab, so send raw text.
        check_embedding_ctx_length=False,
        request_timeout=request_timeout,
        max_retries=0,
    )
