from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from langchain_core.messages import BaseMessage

from .store import ThreadTranscriptStore


@dataclass(frozen=True)
class CheckpointSnapshot:
    messages: list[BaseMessage] = field(default_factory=list)
    state_keys: list[str] = field(default_factory=list)
    message_count: int = 0
    step: int | None = None
    updated_at: str | None = None
    routing_decision: dict[str, object] | None = None
    retrieved_memories: list[dict[str, object]] = field(default_factory=list)


class CheckpointStateReader(Protocol):
    def get_thread_messages(self, thread_id: str, user_id: str = "default") -> list[BaseMessage]:
        ...

    def get_thread_snapshot(self, thread_id: str, user_id: str = "default") -> CheckpointSnapshot:
        ...


class ThreadStateStore(Protocol):
    def record_turn(self, thread_id: str, user_text: str, assistant_text: str) -> None:
        ...

    def get_thread_messages(self, thread_id: str, user_id: str = "default") -> list[BaseMessage]:
        ...

    def get_thread_snapshot(self, thread_id: str, user_id: str = "default") -> CheckpointSnapshot:
        ...


class LangGraphCheckpointStateReader:
    def __init__(self, graph: Any | None = None) -> None:
        self.graph = graph

    def bind_graph(self, graph: Any) -> None:
        self.graph = graph

    def get_thread_messages(self, thread_id: str, user_id: str = "default") -> list[BaseMessage]:
        return self.get_thread_snapshot(thread_id, user_id=user_id).messages

    def get_thread_snapshot(self, thread_id: str, user_id: str = "default") -> CheckpointSnapshot:
        if self.graph is None:
            return CheckpointSnapshot()
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id, "user_id": user_id}})
        values = getattr(snapshot, "values", {}) or {}
        messages = list(values.get("messages", []))
        metadata = getattr(snapshot, "metadata", None) or {}
        updated_at = getattr(snapshot, "created_at", None)
        return CheckpointSnapshot(
            messages=messages,
            state_keys=list(values.keys()),
            message_count=len(messages),
            step=metadata.get("step"),
            updated_at=updated_at,
            routing_decision=values.get("routing_decision"),
            retrieved_memories=list(values.get("retrieved_memories", [])),
        )


class LangGraphThreadStateStore:
    def __init__(
        self,
        checkpoint_state_reader: CheckpointStateReader,
        transcript_store: ThreadTranscriptStore | None = None,
    ) -> None:
        self.checkpoint_state_reader = checkpoint_state_reader
        self.transcript_store = transcript_store

    def record_turn(self, thread_id: str, user_text: str, assistant_text: str) -> None:
        if self.transcript_store is None:
            return
        self.transcript_store.add_message(thread_id, "user", user_text)
        self.transcript_store.add_message(thread_id, "assistant", assistant_text)

    def get_thread_messages(self, thread_id: str, user_id: str = "default") -> list[BaseMessage]:
        return self.checkpoint_state_reader.get_thread_messages(thread_id, user_id=user_id)

    def get_thread_snapshot(self, thread_id: str, user_id: str = "default") -> CheckpointSnapshot:
        return self.checkpoint_state_reader.get_thread_snapshot(thread_id, user_id=user_id)
