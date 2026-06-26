"""Tests for the job store (durable state for the async service)."""

from persistence.job_store import InMemoryJobStore, Job, build_job_store


def test_create_and_get():
    store = InMemoryJobStore()
    store.create(Job(id="j1", prompt="hello"))
    job = store.get("j1")
    assert job is not None
    assert job.status == "queued"
    assert job.prompt == "hello"


def test_update_lifecycle():
    store = InMemoryJobStore()
    store.create(Job(id="j2", prompt="x"))
    store.update("j2", status="running")
    assert store.get("j2").status == "running"
    store.update("j2", status="done", result={"response": "ok", "confidence_score": 0.9})
    job = store.get("j2")
    assert job.status == "done"
    assert job.result["confidence_score"] == 0.9


def test_update_missing_job_is_noop():
    store = InMemoryJobStore()
    store.update("nope", status="done")  # must not raise
    assert store.get("nope") is None


def test_updated_at_changes():
    store = InMemoryJobStore()
    store.create(Job(id="j3"))
    t0 = store.get("j3").updated_at
    store.update("j3", status="running")
    assert store.get("j3").updated_at >= t0


def test_build_default_store_is_in_memory(monkeypatch):
    monkeypatch.delenv("JOB_STORE", raising=False)
    assert isinstance(build_job_store(), InMemoryJobStore)


def test_job_to_dict_roundtrip():
    job = Job(id="j4", prompt="p", status="done", result={"a": 1})
    d = job.to_dict()
    assert d["id"] == "j4" and d["result"] == {"a": 1}
    assert Job(**d).status == "done"
