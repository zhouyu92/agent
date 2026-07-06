from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .sqlite_records import SqliteAuditRepository, SqliteProfileRepository, SqliteTranscriptRepository

if TYPE_CHECKING:
    from .memory import AgentProfile, DedupeEvent, LearningEvent, MemoryEvolutionEvent, RetrievalEvent, RoutingEvent, ThreadMessage


class SqliteProfileStore:
    def __init__(self, db_path: str | Path | object) -> None:
        resolved_path = getattr(db_path, "db_path", db_path)
        self.repository = SqliteProfileRepository(resolved_path)

    def get_profile(self) -> "AgentProfile":
        return self.repository.get_profile()

    def update_profile(
        self,
        *,
        identity: str | None = None,
        style_notes: str | None = None,
        boundaries: str | None = None,
    ) -> None:
        self.repository.update_profile(identity=identity, style_notes=style_notes, boundaries=boundaries)


class SqliteAuditStore:
    def __init__(self, db_path: str | Path | object) -> None:
        resolved_path = getattr(db_path, "db_path", db_path)
        self.repository = SqliteAuditRepository(resolved_path)

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
        self.repository.add_learning_event(
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
    ) -> list["LearningEvent"]:
        return self.repository.recent_learning_events(user_id=user_id, limit=limit, thread_id=thread_id)

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
        self.repository.add_routing_event(
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
    ) -> list["RoutingEvent"]:
        return self.repository.recent_routing_events(
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
        self.repository.add_retrieval_event(
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
    ) -> list["RetrievalEvent"]:
        return self.repository.recent_retrieval_events(user_id=user_id, limit=limit, thread_id=thread_id)

    def add_dedupe_event(
        self,
        *,
        user_id: str,
        thread_id: str | None,
        removed_count: int,
        removed_ids: list[int],
        kept_ids: list[int],
    ) -> None:
        self.repository.add_dedupe_event(
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
    ) -> list["DedupeEvent"]:
        return self.repository.recent_dedupe_events(user_id=user_id, limit=limit, thread_id=thread_id)

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
        self.repository.add_memory_evolution_event(
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
    ) -> list["MemoryEvolutionEvent"]:
        return self.repository.recent_memory_evolution_events(
            user_id=user_id,
            limit=limit,
            thread_id=thread_id,
            action=action,
        )


class SqliteTranscriptStore:
    def __init__(self, db_path: str | Path | object) -> None:
        resolved_path = getattr(db_path, "db_path", db_path)
        self.repository = SqliteTranscriptRepository(resolved_path)

    def add_message(self, thread_id: str, role: str, content: str) -> None:
        self.repository.add_message(thread_id, role, content)

    def recent_messages(self, thread_id: str, limit: int) -> list[dict[str, str]]:
        return self.repository.recent_messages(thread_id, limit)

    def thread_messages(self, thread_id: str, limit: int = 50) -> list["ThreadMessage"]:
        return self.repository.thread_messages(thread_id, limit=limit)
