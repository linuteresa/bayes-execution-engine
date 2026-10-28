"""Versioned collection provenance and atomic experiment manifests.

Manifests describe acquisition; analysis metadata is recorded separately. Historical
rows without this schema remain readable and are explicitly marked as partial provenance.
No API keys or credentials are included.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

SCHEMA_VERSION = 2
CONFIDENCE_SCOPE = "single_step"
ROOT = Path(__file__).resolve().parent.parent


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def experiment_fingerprint(experiment: dict) -> str:
    return digest({
        key: value for key, value in experiment.items()
        if key not in ("experiment_id", "timestamp_utc", "provenance_status", "dataset")
    })


def file_digest(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_digest(*, analysis: bool = False) -> str:
    names = (
        ("eval/metrics.py", "eval/learning.py", "eval/report.py", "eval/grading.py",
         "eval/run_eval.py", "eval/provenance.py", "bayesian_engine/bayes_engine.py",
         "scaling/latent_bayes.py")
        if analysis else (
            "nodes/llm_executor.py", "bayesian_engine/bayes_engine.py", "core/llm.py",
            "core/runtime.py", "core/numbers.py",
            "eval/harness.py", "eval/answerers.py", "eval/datasets.py", "eval/grading.py",
            "eval/provenance.py",
        )
    )
    return digest({name: file_digest(ROOT / name) for name in names if (ROOT / name).exists()})


def public_endpoint(value: str | None) -> str | None:
    if not value:
        return None
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parsed.port is not None:
        host += f":{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def item_identity(item) -> dict:
    return {
        "id": item.id, "question": item.question, "answers": list(item.answers),
        "grader": item.grader, "difficulty": item.difficulty, "source": item.source,
    }


def dependency_versions() -> dict:
    versions = {}
    for package in ("numpy", "scipy", "langchain-openai", "scikit-learn", "datasets", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def inferred_dataset_spec(rows) -> str:
    sources = {r.source for r in rows}
    known = {"triviaqa", "gsm8k", "simpleqa"}
    if sources and sources <= known:
        return "+".join(f"hf:{source}" for source in sorted(sources))
    if sources == {"bundled"}:
        return "bundled"
    return "unknown (stored rows)"


def make_experiment(
    answerer, items, *, n_samples: int, dataset_spec: str | None = None,
    temperature: float | None = None, seed: int | None = None, limit: int | None = None,
    keep_samples: bool = False, similarity: str,
) -> dict:
    from eval.grading import GRADING_VERSION

    source = getattr(answerer, "source_metadata", None)
    simulated = getattr(answerer, "is_simulated", None)
    replay = source is not None
    dataset_spec = (
        source.get("dataset_spec", "unknown (stored rows)") if replay
        else dataset_spec or inferred_dataset_spec(items)
    )
    model = str(getattr(answerer, "model_name", "unknown"))
    sampling = {
        "temperature": temperature if temperature is not None else getattr(answerer, "temperature", None),
        "seed": seed if seed is not None else getattr(answerer, "seed", None),
        "seed_scope": "dataset order and simulator; live server sampling is not seeded",
        "endpoint": public_endpoint(os.getenv("LLAMA_CPP_BASE_URL", "http://localhost:8080/v1"))
        if not replay and simulated is False else None,
        "requested_model": model,
        "server_weights_revision": "not independently recorded" if simulated is False else None,
        "logprobs_requested": bool(getattr(answerer, "request_logprobs", False)),
    }
    config = {
        "schema_version": SCHEMA_VERSION,
        "confidence_scope": CONFIDENCE_SCOPE,
        "answerer": str(getattr(answerer, "name", type(answerer).__name__)),
        "model": model,
        "is_simulated": simulated if simulated in (True, False, None) else None,
        "dataset_spec": dataset_spec,
        "dataset_digest": digest([item_identity(item) for item in items]),
        "limit_per_component": limit if not replay else source.get("limit_per_component"),
        "n_items": len(items),
        "n_samples_per_item": n_samples,
        "sampling": source.get("sampling", {"status": "unknown (legacy rows)"}) if replay else sampling,
        "keep_samples": keep_samples,
        "similarity": similarity,
        "similarity_config": {
            "backend": similarity,
            "endpoint": public_endpoint(os.getenv("EMBEDDINGS_BASE_URL")),
            "model": os.getenv("EMBEDDINGS_MODEL", "local-embeddings")
            if os.getenv("EMBEDDINGS_BASE_URL") else None,
        },
        "execution": {
            "code_sha256": code_digest(), "engine": "fresh_prior_only", "evidence_version": 2,
        },
        "grading_version": GRADING_VERSION,
        "dependencies": dependency_versions(),
        "runtime": {
            "llm_timeout_seconds": os.getenv("LLM_TIMEOUT_SECONDS", "60"),
            "llm_max_tokens": os.getenv("LLM_MAX_TOKENS", "2048"),
        },
    }
    if replay:
        config["sample_source"] = {
            "path": getattr(answerer, "source_path", None),
            "sha256": getattr(answerer, "source_sha256", None),
            "experiment_id": source.get("experiment_id"),
            "provenance_status": source.get("provenance_status", "legacy_partial"),
        }
    return {
        **config,
        "experiment_id": experiment_fingerprint(config),
        "timestamp_utc": utc_now(),
        "provenance_status": "versioned" if not replay or source.get("provenance_status") == "versioned"
        else "legacy_source_partial",
    }


def write_json_atomic(value: dict, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return path


def json_compatible(value):
    """Represent undefined metric values as JSON null, never non-standard NaN tokens."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    return value
