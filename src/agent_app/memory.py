from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .record_stores import SqliteAuditStore, SqliteProfileStore, SqliteTranscriptStore
from .semantic_store import SqliteSemanticMemoryStore
from .sqlite_schema import ensure_sqlite_schema
from .storage_utils import clean_user_id, redact_sensitive

if TYPE_CHECKING:
    from .semantic_store import VectorMemoryIndexer, VectorMemorySearcher


@dataclass(frozen=True)
class MemoryItem:
    id: int
    category: str
    content: str
    importance: int
    source: str
    created_at: str
    status: str = "active"
    last_confirmed_at: str | None = None


@dataclass(frozen=True)
class AgentProfile:
    identity: str
    style_notes: str
    boundaries: str
    updated_at: str


@dataclass(frozen=True)
class LearningEvent:
    id: int
    user_id: str
    thread_id: str
    user_text: str
    assistant_text: str
    memory_count: int
    profile_fields: list[str]
    created_at: str


@dataclass(frozen=True)
class ReflectionEvent:
    id: int
    user_id: str
    thread_id: str
    source_event_ids: list[int]
    summary: str
    memory_count: int
    profile_fields: list[str]
    created_at: str


@dataclass(frozen=True)
class RoutingEvent:
    id: int
    user_id: str
    thread_id: str
    user_text: str
    should_retrieve: bool
    retrieve_reason: str
    should_learn: bool
    learn_reason: str
    created_at: str


@dataclass(frozen=True)
class ThreadMessage:
    id: int
    thread_id: str
    role: str
    content: str
    created_at: str


@dataclass(frozen=True)
class RetrievalEvent:
    id: int
    user_id: str
    thread_id: str
    user_text: str
    memory_count: int
    memory_ids: list[int]
    memory_preview: str
    created_at: str


@dataclass(frozen=True)
class DedupeResult:
    removed_count: int
    removed_ids: list[int]
    kept_ids: list[int]


@dataclass(frozen=True)
class DedupeEvent:
    id: int
    user_id: str
    thread_id: str | None
    removed_count: int
    removed_ids: list[int]
    kept_ids: list[int]
    created_at: str


@dataclass(frozen=True)
class MemoryEvolutionEvent:
    id: int
    user_id: str
    thread_id: str | None
    action: str
    candidate_category: str
    candidate_content: str
    target_memory_id: int | None
    result_memory_id: int | None
    reason: str
    created_at: str


@dataclass(frozen=True)
class MemoryEvolutionResult:
    action: str
    candidate_category: str
    candidate_content: str
    target_memory_id: int | None
    result_memory_id: int | None
    reason: str


class MemoryStore:
    def __init__(
        self,
        db_path: str | Path,
        *,
        vector_indexer: "VectorMemoryIndexer | None" = None,
        vector_searcher: "VectorMemorySearcher | None" = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        ensure_sqlite_schema(self.db_path)
        self.semantic_store = SqliteSemanticMemoryStore(
            self.db_path,
            vector_indexer=vector_indexer,
            vector_searcher=vector_searcher,
        )
        self.profile_store = SqliteProfileStore(self.db_path)
        self.audit_store = SqliteAuditStore(self.db_path)
        self.transcript_store = SqliteTranscriptStore(self.db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add_memory(
        self,
        category: str,
        content: str,
        importance: int,
        source: str,
        user_id: str = "default",
    ) -> bool:
        return self.semantic_store.add_memory(category, content, importance, source, user_id=user_id)

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
        result = self.semantic_store.evolve_memory(
            category=category,
            content=content,
            importance=importance,
            source=source,
            user_id=user_id,
            thread_id=thread_id,
        )
        self.add_memory_evolution_event(
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

    def search_memories(self, query: str, limit: int = 5, user_id: str = "default", status: str = "active") -> list[MemoryItem]:
        if status == "active":
            return self.semantic_store.search_memories(query, limit=limit, user_id=user_id)
        return self.semantic_store.search_memories(query, limit=limit, user_id=user_id, status=status)

    def recent_memories(self, limit: int = 10, user_id: str = "default", status: str = "active") -> list[MemoryItem]:
        if status == "active":
            return self.semantic_store.recent_memories(limit=limit, user_id=user_id)
        return self.semantic_store.recent_memories(limit=limit, user_id=user_id, status=status)

    def delete_memory(self, memory_id: int, user_id: str = "default") -> bool:
        return self.semantic_store.delete_memory(memory_id, user_id=user_id)

    def archive_memory(self, memory_id: int, user_id: str = "default") -> bool:
        return self.semantic_store.archive_memory(memory_id, user_id=user_id)

    def restore_memory(self, memory_id: int, user_id: str = "default") -> bool:
        row = self.semantic_store.repository.archived_memory_by_id(user_id, memory_id)
        if row is None:
            return False
        restored = self.semantic_store.restore_memory(memory_id, user_id=user_id)
        if not restored:
            return False
        self.add_memory_evolution_event(
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
        row = self.semantic_store.repository.active_memory_by_id(user_id, memory_id)
        if row is None:
            return False
        confirmed = self.semantic_store.confirm_memory(memory_id, user_id=user_id)
        if not confirmed:
            return False
        self.add_memory_evolution_event(
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
        result = self.semantic_store.dedupe_memories(user_id=user_id)
        self.audit_store.add_dedupe_event(
            user_id=user_id,
            thread_id=thread_id,
            removed_count=result.removed_count,
            removed_ids=result.removed_ids,
            kept_ids=result.kept_ids,
        )
        return result

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

    def recent_learning_events(
        self,
        user_id: str = "default",
        limit: int = 10,
        *,
        thread_id: str | None = None,
    ) -> list[LearningEvent]:
        return self.audit_store.recent_learning_events(user_id=user_id, limit=limit, thread_id=thread_id)

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
        return self.audit_store.recent_routing_events(
            user_id=user_id,
            limit=limit,
            thread_id=thread_id,
            learn=learn,
            retrieve=retrieve,
            reason=reason,
            text_query=text_query,
        )

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

    def recent_retrieval_events(
        self,
        user_id: str = "default",
        limit: int = 10,
        *,
        thread_id: str | None = None,
    ) -> list[RetrievalEvent]:
        return self.audit_store.recent_retrieval_events(user_id=user_id, limit=limit, thread_id=thread_id)

    def add_dedupe_event(
        self,
        *,
        user_id: str,
        thread_id: str | None = None,
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
        thread_id: str | None = None,
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

    def add_message(self, thread_id: str, role: str, content: str) -> None:
        self.transcript_store.add_message(thread_id, role, content)

    def recent_messages(self, thread_id: str, limit: int) -> list[dict[str, str]]:
        return self.transcript_store.recent_messages(thread_id, limit)

    def thread_messages(self, thread_id: str, limit: int = 50) -> list[ThreadMessage]:
        return self.transcript_store.thread_messages(thread_id, limit)

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
