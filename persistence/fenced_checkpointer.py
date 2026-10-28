"""Attempt-isolated checkpoints with an ownership-fenced recovery reference.

A saver write and a job-store transaction cannot commit together. Each attempt
therefore writes its own physical thread, then publishes an exact checkpoint ID
through the job store's lease check. Recovery copies only that published full
checkpoint to a new attempt thread. Unpublished writes and pending task writes
from an old worker never become recovery input.
"""

from __future__ import annotations

import hashlib
import threading
from copy import deepcopy
from typing import Any, Callable, Iterator, Sequence

from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple, copy_checkpoint

from core.runtime import ExecutionCancelled


def checkpoint_reference(config: dict[str, Any]) -> dict[str, Any]:
    """Keep the exact saver address, excluding model clients and request secrets."""
    settings = config.get("configurable", {})
    thread_id = settings.get("thread_id")
    checkpoint_id = settings.get("checkpoint_id")
    namespace = settings.get("checkpoint_ns", "")
    if (not isinstance(thread_id, str) or not thread_id
            or not isinstance(checkpoint_id, str) or not checkpoint_id
            or not isinstance(namespace, str)):
        raise ValueError("checkpoint reference requires exact thread, namespace and checkpoint IDs")
    return {"configurable": {"thread_id": thread_id, "checkpoint_ns": namespace,
                             "checkpoint_id": checkpoint_id}}


class FencedCheckpointer(BaseCheckpointSaver):
    """Publish completed saver writes only while the current attempt owns its job.

    ``publish_callback(reference)`` must atomically verify the job's current owner
    and unexpired lease before recording the reference, returning ``True`` on
    success. A new wrapper and a new physical thread ID are required per attempt.
    This is a synchronous saver for the synchronous execution graph.
    """

    def __init__(self, delegate: BaseCheckpointSaver,
                 publish_callback: Callable[[dict[str, Any]], bool]) -> None:
        if not callable(publish_callback):
            raise ValueError("publish_callback must be callable")
        super().__init__(serde=delegate.serde)
        self.delegate = delegate
        self.publish_callback = publish_callback
        self._lock = threading.RLock()
        self._thread_id: str | None = None
        self._cancelled = False
        self._checkpoint_hashes: dict[tuple[str, str], bytes] = {}

    @property
    def config_specs(self):
        return self.delegate.config_specs

    def _destination(self, config: dict[str, Any]) -> None:
        thread_id = config.get("configurable", {}).get("thread_id")
        if not isinstance(thread_id, str) or not thread_id:
            raise ValueError("attempt requires a nonempty checkpoint thread ID")
        if self._thread_id is None:
            self._thread_id = thread_id
        elif self._thread_id != thread_id:
            raise ValueError("one fenced saver cannot write multiple attempt threads")
        if self._cancelled:
            raise ExecutionCancelled("Checkpoint publication rejected after worker lease loss")

    def get_tuple(self, config: dict[str, Any]) -> CheckpointTuple | None:
        return self.delegate.get_tuple(config)

    def list(self, config: dict[str, Any] | None, *, filter=None, before=None,
             limit=None) -> Iterator[CheckpointTuple]:
        return self.delegate.list(config, filter=filter, before=before, limit=limit)

    def put(self, config, checkpoint, metadata, new_versions):
        with self._lock:
            self._destination(config)
            # An exact published ID must continue to identify the same full state.
            # LangGraph may re-save an identical checkpoint when its loop exits.
            key = (config["configurable"].get("checkpoint_ns", ""), checkpoint["id"])
            serialized_type, serialized = self.serde.dumps_typed(checkpoint)
            digest = hashlib.sha256(serialized_type.encode() + serialized).digest()
            previous = self._checkpoint_hashes.get(key)
            if previous is not None and previous != digest:
                raise ValueError("cannot overwrite a checkpoint ID with different state")
            saved = self.delegate.put(config, checkpoint, metadata, new_versions)
            reference = checkpoint_reference(saved)
            if reference["configurable"]["thread_id"] != self._thread_id:
                raise ValueError("saver returned a different attempt thread")
            self._checkpoint_hashes[key] = digest
            try:
                accepted = self.publish_callback(deepcopy(reference))
            except Exception as exc:
                self._cancelled = True
                raise ExecutionCancelled("Checkpoint ownership could not be confirmed") from exc
            if not accepted:
                self._cancelled = True
                raise ExecutionCancelled("Checkpoint publication rejected after worker lease loss")
            return saved

    def put_writes(self, config, writes: Sequence[tuple[str, Any]], task_id: str,
                   task_path: str = "") -> None:
        with self._lock:
            self._destination(config)
            self.delegate.put_writes(config, writes, task_id, task_path)

    def get_next_version(self, current, channel):
        return self.delegate.get_next_version(current, channel)

    def delete_thread(self, thread_id: str) -> None:
        self.delegate.delete_thread(thread_id)

    def restore_from(self, reference: dict[str, Any], new_config: dict[str, Any]) -> dict[str, Any]:
        """Clone an exact committed checkpoint, ignoring all pending task writes.

        Missing referenced checkpoints are errors; choosing a newer checkpoint or
        silently restarting would violate the persisted recovery boundary.
        """
        reference = checkpoint_reference(reference)
        destination = {**new_config, "configurable": {
            **new_config.get("configurable", {}),
            "checkpoint_ns": reference["configurable"]["checkpoint_ns"],
        }}
        destination["configurable"].pop("checkpoint_id", None)
        if destination["configurable"].get("thread_id") == reference["configurable"]["thread_id"]:
            raise ValueError("recovery requires a new attempt checkpoint thread")
        saved = self.delegate.get_tuple(reference)
        if saved is None:
            raise ValueError("the committed recovery checkpoint is unavailable")
        if saved.checkpoint["id"] != reference["configurable"]["checkpoint_id"]:
            raise ValueError("saver did not return the exact committed checkpoint")
        checkpoint = deepcopy(copy_checkpoint(saved.checkpoint))
        metadata = deepcopy(saved.metadata)
        metadata["source"] = "fork"
        # Save every channel blob under the new physical thread. Merely copying
        # the checkpoint record would leave the referenced versions unavailable.
        return self.put(destination, checkpoint, metadata, checkpoint["channel_versions"].copy())


# Name retained for callers that describe the wrapper by its isolation boundary.
AttemptCheckpointer = FencedCheckpointer
