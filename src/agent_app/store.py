from __future__ import annotations

import sqlite3
from typing import Protocol, runtime_checkable

from .memory import (
    AgentProfile,
    DedupeEvent,
    DedupeResult,
    LearningEvent,
    MemoryEvolutionResult,
    MemoryEvolutionEvent,
    MemoryItem,
    MemoryStore,
    ReflectionEvent,
    RetrievalEvent,
    RoutingEvent,
    ThreadMessage,
)
from .record_stores import SqliteAuditStore, SqliteProfileStore, SqliteTranscriptStore
from .semantic_store import SqliteSemanticMemoryStore, VectorMemoryIndexer, VectorMemorySearcher


@runtime_checkable
class SemanticMemoryStore(Protocol):
    def search_memories(self, query: str, limit: int = 5, user_id: str = "default") -> list[MemoryItem]:
        ...

    def add_memory(
        self,
        category: str,
        content: str,
        importance: int,
        source: str,
        user_id: str = "default",
    ) -> bool:
        ...

    def dedupe_memories(self, user_id: str = "default") -> DedupeResult:
        ...

    def evolve_memory(
        self,
        *,
        category: str,
        content: str,
        importance: int,
        source: str,
        user_id: str = "default",
        thread_id: str | None = None,
    ) -> MemoryEvolutionResult:
        ...


@runtime_checkable
class MemoryProfileStore(Protocol):
    def get_profile(self) -> AgentProfile:
        ...

    def update_profile(
        self,
        *,
        identity: str | None = None,
        style_notes: str | None = None,
        boundaries: str | None = None,
    ) -> None:
        ...


@runtime_checkable
class AuditStore(Protocol):
    def add_learning_event(
        self,
        *,
        user_id: str,
        thread_id: str,
        user_text: str,
        assistant_text: str,
        memory_count: int,
        profile_fields: list[str],
    ) -> None:
        ...

    def add_routing_event(
        self,
        *,
        user_id: str,
        thread_id: str,
        user_text: str,
        should_retrieve: bool,
        retrieve_reason: str,
        should_learn: bool,
        learn_reason: str,
    ) -> None:
        ...

    def add_reflection_event(
        self,
        *,
        user_id: str,
        thread_id: str,
        source_event_ids: list[int],
        summary: str,
        memory_count: int,
        profile_fields: list[str],
    ) -> None:
        ...

    def recent_reflection_events(
        self,
        user_id: str = "default",
        limit: int = 10,
        *,
        thread_id: str | None = None,
    ) -> list[ReflectionEvent]:
        ...

    def add_retrieval_event(
        self,
        *,
        user_id: str,
        thread_id: str,
        user_text: str,
        memory_count: int,
        memory_ids: list[int],
        memory_preview: str,
    ) -> None:
        ...

    def add_dedupe_event(
        self,
        *,
        user_id: str,
        thread_id: str | None,
        removed_count: int,
        removed_ids: list[int],
        kept_ids: list[int],
    ) -> None:
        ...

    def recent_dedupe_events(
        self,
        user_id: str = "default",
        limit: int = 10,
        *,
        thread_id: str | None = None,
    ) -> list[DedupeEvent]:
        ...

    def add_memory_evolution_event(
        self,
        *,
        user_id: str,
        thread_id: str | None,
        action: str,
        candidate_category: str,
        candidate_content: str,
        target_memory_id: int | None,
        result_memory_id: int | None,
        reason: str,
    ) -> None:
        ...

    def recent_memory_evolution_events(
        self,
        user_id: str = "default",
        limit: int = 10,
        *,
        thread_id: str | None = None,
        action: str | None = None,
        reason: str | None = None,
    ) -> list[MemoryEvolutionEvent]:
        ...


class LongTermStore(SemanticMemoryStore, MemoryProfileStore, AuditStore, Protocol):
    pass


@runtime_checkable
class ThreadTranscriptStore(Protocol):
    def add_message(self, thread_id: str, role: str, content: str) -> None:
        ...


class SqliteLongTermStore:
    def __init__(
        self,
        memory: MemoryStore,
        vector_indexer: VectorMemoryIndexer | None = None,
        vector_searcher: VectorMemorySearcher | None = None,
    ) -> None:
        self.memory = memory
        self.semantic_memory_store = SqliteSemanticMemoryStore(
            memory.db_path,
            vector_indexer=vector_indexer,
            vector_searcher=vector_searcher,
        )
        self.profile_store = SqliteProfileStore(memory)
        self.audit_store = SqliteAuditStore(memory)
        self.transcript_store = SqliteTranscriptStore(memory)

    def search_memories(self, query: str, limit: int = 5, user_id: str = "default") -> list[MemoryItem]:
        return self.semantic_memory_store.search_memories(query, limit=limit, user_id=user_id)

    def add_memory(
        self,
        category: str,
        content: str,
        importance: int,
        source: str,
        user_id: str = "default",
    ) -> bool:
        return self.semantic_memory_store.add_memory(
            category,
            content,
            importance,
            source,
            user_id=user_id,
        )

    def evolve_memory(
        self,
        *,
        category: str,
        content: str,
        importance: int,
        source: str,
        user_id: str = "default",
        thread_id: str | None = None,
    ) -> MemoryEvolutionResult:
        result = self.semantic_memory_store.evolve_memory(
            category=category,
            content=content,
            importance=importance,
            source=source,
            user_id=user_id,
            thread_id=thread_id,
        )
        self.audit_store.add_memory_evolution_event(
            user_id=user_id,
            thread_id=thread_id,
            action=result.action,
            candidate_category=result.candidate_category,
            candidate_content=result.candidate_content,
            target_memory_id=result.target_memory_id,
            result_memory_id=result.result_memory_id,
            reason=result.reason,
        )
        return result

    def dedupe_memories(self, user_id: str = "default", thread_id: str | None = None) -> DedupeResult:
        result = self.semantic_memory_store.dedupe_memories(user_id=user_id)
        self.audit_store.add_dedupe_event(
            user_id=user_id,
            thread_id=thread_id,
            removed_count=result.removed_count,
            removed_ids=result.removed_ids,
            kept_ids=result.kept_ids,
        )
        return result

    def get_profile(self) -> AgentProfile:
        return self.profile_store.get_profile()

    def update_profile(
        self,
        *,
        identity: str | None = None,
        style_notes: str | None = None,
        boundaries: str | None = None,
    ) -> None:
        self.profile_store.update_profile(identity=identity, style_notes=style_notes, boundaries=boundaries)

    def add_learning_event(
        self,
        *,
        user_id: str,
        thread_id: str,
        user_text: str,
        assistant_text: str,
        memory_count: int,
        profile_fields: list[str],
    ) -> None:
        self.audit_store.add_learning_event(
            user_id=user_id,
            thread_id=thread_id,
            user_text=user_text,
            assistant_text=assistant_text,
            memory_count=memory_count,
            profile_fields=profile_fields,
        )

    def add_routing_event(
        self,
        *,
        user_id: str,
        thread_id: str,
        user_text: str,
        should_retrieve: bool,
        retrieve_reason: str,
        should_learn: bool,
        learn_reason: str,
    ) -> None:
        self.audit_store.add_routing_event(
            user_id=user_id,
            thread_id=thread_id,
            user_text=user_text,
            should_retrieve=should_retrieve,
            retrieve_reason=retrieve_reason,
            should_learn=should_learn,
            learn_reason=learn_reason,
        )

    def add_reflection_event(
        self,
        *,
        user_id: str,
        thread_id: str,
        source_event_ids: list[int],
        summary: str,
        memory_count: int,
        profile_fields: list[str],
    ) -> None:
        self.audit_store.add_reflection_event(
            user_id=user_id,
            thread_id=thread_id,
            source_event_ids=source_event_ids,
            summary=summary,
            memory_count=memory_count,
            profile_fields=profile_fields,
        )

    def recent_reflection_events(
        self,
        user_id: str = "default",
        limit: int = 10,
        *,
        thread_id: str | None = None,
    ) -> list[ReflectionEvent]:
        return self.audit_store.recent_reflection_events(user_id=user_id, limit=limit, thread_id=thread_id)

    def add_retrieval_event(
        self,
        *,
        user_id: str,
        thread_id: str,
        user_text: str,
        memory_count: int,
        memory_ids: list[int],
        memory_preview: str,
    ) -> None:
        self.audit_store.add_retrieval_event(
            user_id=user_id,
            thread_id=thread_id,
            user_text=user_text,
            memory_count=memory_count,
            memory_ids=memory_ids,
            memory_preview=memory_preview,
        )

    def add_dedupe_event(
        self,
        *,
        user_id: str,
        thread_id: str | None,
        removed_count: int,
        removed_ids: list[int],
        kept_ids: list[int],
    ) -> None:
        self.audit_store.add_dedupe_event(
            user_id=user_id,
            thread_id=thread_id,
            removed_count=removed_count,
            removed_ids=removed_ids,
            kept_ids=kept_ids,
        )

    def recent_dedupe_events(
        self,
        user_id: str = "default",
        limit: int = 10,
        *,
        thread_id: str | None = None,
    ) -> list[DedupeEvent]:
        return self.audit_store.recent_dedupe_events(user_id=user_id, limit=limit, thread_id=thread_id)

    def add_memory_evolution_event(
        self,
        *,
        user_id: str,
        thread_id: str | None,
        action: str,
        candidate_category: str,
        candidate_content: str,
        target_memory_id: int | None,
        result_memory_id: int | None,
        reason: str,
    ) -> None:
        self.audit_store.add_memory_evolution_event(
            user_id=user_id,
            thread_id=thread_id,
            action=action,
            candidate_category=candidate_category,
            candidate_content=candidate_content,
            target_memory_id=target_memory_id,
            result_memory_id=result_memory_id,
            reason=reason,
        )

    def recent_memory_evolution_events(
        self,
        user_id: str = "default",
        limit: int = 10,
        *,
        thread_id: str | None = None,
        action: str | None = None,
        reason: str | None = None,
    ) -> list[MemoryEvolutionEvent]:
        return self.audit_store.recent_memory_evolution_events(
            user_id=user_id,
            limit=limit,
            thread_id=thread_id,
            action=action,
            reason=reason,
        )


class SqliteCliStore:
    def __init__(self, memory: MemoryStore, long_term_store: SqliteLongTermStore | None = None) -> None:
        self.memory = memory
        self.long_term_store = long_term_store or SqliteLongTermStore(memory)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.memory.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_profile(self) -> AgentProfile:
        return self.long_term_store.profile_store.get_profile()

    def search_memories(self, query: str, limit: int = 10, user_id: str = "default", status: str = "active") -> list[MemoryItem]:
        if status == "active":
            return self.long_term_store.semantic_memory_store.search_memories(query, limit=limit, user_id=user_id)
        return self.long_term_store.semantic_memory_store.search_memories(query, limit=limit, user_id=user_id, status=status)

    def recent_memories(self, limit: int = 10, user_id: str = "default", status: str = "active") -> list[MemoryItem]:
        if status == "active":
            return self.long_term_store.semantic_memory_store.recent_memories(limit=limit, user_id=user_id)
        return self.long_term_store.semantic_memory_store.recent_memories(limit=limit, user_id=user_id, status=status)

    def delete_memory(self, memory_id: int, user_id: str = "default") -> bool:
        return self.long_term_store.semantic_memory_store.delete_memory(memory_id, user_id=user_id)

    def archive_memory(self, memory_id: int, user_id: str = "default") -> bool:
        return self.long_term_store.semantic_memory_store.archive_memory(memory_id, user_id=user_id)

    def restore_memory(self, memory_id: int, user_id: str = "default") -> bool:
        row = self.long_term_store.semantic_memory_store.repository.archived_memory_by_id(user_id, memory_id)
        if row is None:
            return False
        restored = self.long_term_store.semantic_memory_store.restore_memory(memory_id, user_id=user_id)
        if not restored:
            return False
        self.long_term_store.audit_store.add_memory_evolution_event(
            user_id=user_id,
            thread_id=None,
            action="restore",
            candidate_category=row["category"],
            candidate_content=row["content"],
            target_memory_id=memory_id,
            result_memory_id=memory_id,
            reason="manual_restore",
        )
        return True

    def confirm_memory(self, memory_id: int, user_id: str = "default") -> bool:
        row = self.long_term_store.semantic_memory_store.repository.active_memory_by_id(user_id, memory_id)
        if row is None:
            return False
        confirmed = self.long_term_store.semantic_memory_store.confirm_memory(memory_id, user_id=user_id)
        if not confirmed:
            return False
        self.long_term_store.audit_store.add_memory_evolution_event(
            user_id=user_id,
            thread_id=None,
            action="reinforce",
            candidate_category=row["category"],
            candidate_content=row["content"],
            target_memory_id=memory_id,
            result_memory_id=memory_id,
            reason="manual_confirmation",
        )
        return True

    def dedupe_memories(self, user_id: str = "default", thread_id: str | None = None) -> DedupeResult:
        result = self.long_term_store.dedupe_memories(user_id=user_id, thread_id=thread_id)
        return result

    def recent_learning_events(
        self,
        user_id: str = "default",
        limit: int = 10,
        *,
        thread_id: str | None = None,
    ) -> list[LearningEvent]:
        return self.long_term_store.audit_store.recent_learning_events(
            user_id=user_id,
            limit=limit,
            thread_id=thread_id,
        )

    def recent_reflection_events(
        self,
        user_id: str = "default",
        limit: int = 10,
        *,
        thread_id: str | None = None,
    ) -> list[ReflectionEvent]:
        return self.long_term_store.recent_reflection_events(user_id=user_id, limit=limit, thread_id=thread_id)

    def recent_routing_events(
        self,
        user_id: str = "default",
        limit: int = 10,
        *,
        thread_id: str | None = None,
        learn: bool | None = None,
        retrieve: bool | None = None,
        reason: str | None = None,
        text_query: str | None = None,
    ) -> list[RoutingEvent]:
        return self.long_term_store.audit_store.recent_routing_events(
            user_id=user_id,
            limit=limit,
            thread_id=thread_id,
            learn=learn,
            retrieve=retrieve,
            reason=reason,
            text_query=text_query,
        )

    def recent_retrieval_events(
        self,
        user_id: str = "default",
        limit: int = 10,
        *,
        thread_id: str | None = None,
    ) -> list[RetrievalEvent]:
        return self.long_term_store.audit_store.recent_retrieval_events(
            user_id=user_id,
            limit=limit,
            thread_id=thread_id,
        )

    def recent_dedupe_events(
        self,
        user_id: str = "default",
        limit: int = 10,
        *,
        thread_id: str | None = None,
    ) -> list[DedupeEvent]:
        return self.long_term_store.audit_store.recent_dedupe_events(
            user_id=user_id,
            limit=limit,
            thread_id=thread_id,
        )

    def recent_memory_evolution_events(
        self,
        user_id: str = "default",
        limit: int = 10,
        *,
        thread_id: str | None = None,
        action: str | None = None,
        reason: str | None = None,
    ) -> list[MemoryEvolutionEvent]:
        return self.long_term_store.audit_store.recent_memory_evolution_events(
            user_id=user_id,
            limit=limit,
            thread_id=thread_id,
            action=action,
            reason=reason,
        )

    def thread_messages(self, thread_id: str, limit: int = 50) -> list[ThreadMessage]:
        return self.long_term_store.transcript_store.thread_messages(thread_id, limit=limit)
