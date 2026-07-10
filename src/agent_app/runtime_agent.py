from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from .agent import ConversationalAgent
from .memory import DedupeEvent, LearningEvent, MemoryEvolutionEvent, ReflectionEvent, RetrievalEvent, RoutingEvent, ThreadMessage
from .reflection import ReflectionRunResult


@runtime_checkable
class ConversationRuntime(Protocol):
    def reply(self, user_text: str, thread_id: str = "default", user_id: str = "default") -> str:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class ThreadInspection:
    thread_id: str
    transcript_messages: list[ThreadMessage] = field(default_factory=list)
    routing_events: list[RoutingEvent] = field(default_factory=list)
    retrieval_events: list[RetrievalEvent] = field(default_factory=list)
    learning_events: list[LearningEvent] = field(default_factory=list)
    reflection_events: list[ReflectionEvent] = field(default_factory=list)
    dedupe_events: list[DedupeEvent] = field(default_factory=list)
    memory_evolution_events: list[MemoryEvolutionEvent] = field(default_factory=list)
    checkpoint_available: bool = False
    checkpoint_state_keys: list[str] = field(default_factory=list)
    checkpoint_message_count: int = 0
    checkpoint_step: int | None = None
    checkpoint_updated_at: str | None = None
    checkpoint_routing_decision: dict[str, object] | None = None
    checkpoint_retrieved_memories: list[dict[str, object]] = field(default_factory=list)
    checkpoint_messages: list[object] = field(default_factory=list)


@runtime_checkable
class ThreadInspectionRuntime(Protocol):
    def inspect_thread(self, thread_id: str, user_id: str = "default") -> ThreadInspection:
        ...


@runtime_checkable
class ReflectionRuntime(Protocol):
    def reflect(
        self,
        thread_id: str = "default",
        user_id: str = "default",
        min_episode_count: int = 2,
    ) -> ReflectionRunResult:
        ...


class ClassicConversationRuntime:
    def __init__(self, agent: ConversationalAgent, cli_store) -> None:
        self.agent = agent
        self.cli_store = cli_store

    def reply(self, user_text: str, thread_id: str = "default", user_id: str = "default") -> str:
        return self.agent.reply(user_text, thread_id=thread_id, user_id=user_id)

    def reflect(
        self,
        thread_id: str = "default",
        user_id: str = "default",
        min_episode_count: int = 2,
    ) -> ReflectionRunResult:
        return self.agent.reflect(
            thread_id=thread_id,
            user_id=user_id,
            min_episode_count=min_episode_count,
        )

    def summarize_thread(self, thread_id: str, user_id: str = "default") -> str | None:
        return self.agent.summarize_thread(thread_id, user_id=user_id)

    def get_thread_messages(self, thread_id: str, user_id: str = "default") -> list[ThreadMessage]:
        return []

    def inspect_thread(self, thread_id: str, user_id: str = "default") -> ThreadInspection:
        return ThreadInspection(
            thread_id=thread_id,
            transcript_messages=self.cli_store.thread_messages(thread_id, limit=50),
            routing_events=self.cli_store.recent_routing_events(user_id=user_id, limit=20, thread_id=thread_id),
            retrieval_events=self.cli_store.recent_retrieval_events(user_id=user_id, limit=20, thread_id=thread_id),
            learning_events=self.cli_store.recent_learning_events(user_id=user_id, limit=20, thread_id=thread_id),
            reflection_events=self.cli_store.recent_reflection_events(user_id=user_id, limit=20, thread_id=thread_id),
            dedupe_events=self.cli_store.recent_dedupe_events(user_id=user_id, limit=20, thread_id=thread_id),
            memory_evolution_events=self.cli_store.recent_memory_evolution_events(
                user_id=user_id, limit=20, thread_id=thread_id
            ),
            checkpoint_available=False,
            checkpoint_messages=self.get_thread_messages(thread_id, user_id=user_id),
        )

    def close(self) -> None:
        return None
