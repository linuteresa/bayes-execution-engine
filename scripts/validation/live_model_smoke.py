"""Kill/restart the dedicated validation container after a real model step commits.

Requires the validation API on localhost:18000 and Postgres on localhost:15432.
This script controls only the explicitly named, disposable validation container.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import httpx

from persistence.checkpointer import build_checkpointer

CONTAINER = "bayes-engine-model-validation"


def wait_for(predicate, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.05)
    raise TimeoutError("live model recovery did not complete")


def main():
    os.environ["CHECKPOINTER"] = "postgres"
    os.environ.setdefault(
        "POSTGRES_DSN",
        "postgresql://bayes_test:local-validation-only@127.0.0.1:15432/bayes_test",
    )
    with httpx.Client(base_url="http://127.0.0.1:18000", timeout=5, trust_env=False,
                      headers={"Authorization": "Bearer local-validation-only"}) as client:
        accepted = client.post("/jobs", json={
            "question": "Identify the capital of France, then multiply 17 by 19. "
                        "Use separate steps and give both answers in the final response.",
        })
        accepted.raise_for_status()
        job_id = accepted.json()["job_id"]

        def read_job():
            response = client.get(f"/jobs/{job_id}")
            response.raise_for_status()
            job = response.json()
            if job["status"] == "error":
                raise RuntimeError(f"model workflow failed: {job}")
            return job

        with build_checkpointer() as saver:
            def committed_step():
                job = read_job()
                if job["status"] == "done":
                    raise RuntimeError("workflow finished before crash injection; repeat the smoke test")
                if not job["checkpoint_ref"]:
                    return None
                saved = saver.get_tuple(job["checkpoint_ref"])
                state = saved.checkpoint["channel_values"]
                return job if state.get("past_steps") and not state.get("response") else None

            before = wait_for(committed_step)
        print(f"Saved real execution step for {job_id}; killing validation container", flush=True)
        subprocess.run(["docker", "kill", CONTAINER], check=True)
        subprocess.run(["docker", "start", CONTAINER], check=True)

        def completed():
            try:
                job = read_job()
            except httpx.TransportError:
                return None
            return job if job["status"] == "done" else None

        result = wait_for(completed)
        assert result["attempts"] == 2, result
        assert result["checkpoint_ref"] != before["checkpoint_ref"]
        assert "paris" in result["result"]["response"].lower(), result
        assert "323" in result["result"]["response"], result
        feedback = client.post(f"/jobs/{job_id}/feedback", json={"correct": True})
        feedback.raise_for_status()
        assert client.post(f"/jobs/{job_id}/feedback", json={"correct": True}).status_code == 409
        subprocess.run(["docker", "restart", CONTAINER], check=True)
        recovered = wait_for(completed)
        assert recovered["result"] == result["result"] and recovered["feedback"] is True
        out = Path("data/live-model-recovery.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"before_crash": before, "recovered": recovered,
                                   "feedback": feedback.json()}, indent=2) + "\n")
        print(f"Recovery and feedback survived two restarts; evidence: {out}", flush=True)


if __name__ == "__main__":
    main()
