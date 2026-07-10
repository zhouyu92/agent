from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import TYPE_CHECKING

from .storage_utils import clean_user_id, now_iso, redact_sensitive

if TYPE_CHECKING:
    from .memory import AgentProfile, DedupeEvent, LearningEvent, MemoryEvolutionEvent, ReflectionEvent, RetrievalEvent, RoutingEvent, ThreadMessage


class SqliteProfileRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_profile(self) -> "AgentProfile":
        from .memory import AgentProfile

        with self._connect() as conn:
            row = conn.execute("SELECT * FROM agent_profile WHERE id = 1").fetchone()
        return AgentProfile(
            identity=row["identity"],
            style_notes=row["style_notes"],
            boundaries=row["boundaries"],
            updated_at=row["updated_at"],
        )

    def update_profile(
        self,
        *,
        identity: str | None = None,
        style_notes: str | None = None,
        boundaries: str | None = None,
    ) -> None:
        current = self.get_profile()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE agent_profile
                SET identity = ?, style_notes = ?, boundaries = ?, updated_at = ?
                WHERE id = 1
                """,
                (
                    identity if identity is not None else current.identity,
                    style_notes if style_notes is not None else current.style_notes,
                    boundaries if boundaries is not None else current.boundaries,
                    now_iso(),
                ),
            )


class SqliteAuditRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

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
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO learning_events (
                    user_id, thread_id, user_text, assistant_text,
                    memory_count, profile_fields, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_user_id(user_id),
                    thread_id,
                    redact_sensitive(user_text),
                    redact_sensitive(assistant_text),
                    memory_count,
                    ",".join(profile_fields),
                    now_iso(),
                ),
            )

    def recent_learning_events(
        self,
        user_id: str = "default",
        limit: int = 10,
        *,
        thread_id: str | None = None,
    ) -> list["LearningEvent"]:
        from .memory import LearningEvent

        clauses = ["user_id = ?"]
        params: list[object] = [clean_user_id(user_id)]
        if thread_id:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM learning_events
                WHERE {' AND '.join(clauses)}
                ORDER BY id DESC
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
        return [
            LearningEvent(
                id=row["id"],
                user_id=row["user_id"],
                thread_id=row["thread_id"],
                user_text=row["user_text"],
                assistant_text=row["assistant_text"],
                memory_count=row["memory_count"],
                profile_fields=[field for field in row["profile_fields"].split(",") if field],
                created_at=row["created_at"],
            )
            for row in rows
        ]

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
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO reflection_events (
                    user_id, thread_id, source_event_ids, summary,
                    memory_count, profile_fields, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_user_id(user_id),
                    thread_id,
                    ",".join(str(event_id) for event_id in source_event_ids),
                    redact_sensitive(summary),
                    memory_count,
                    ",".join(profile_fields),
                    now_iso(),
                ),
            )

    def recent_reflection_events(
        self,
        user_id: str = "default",
        limit: int = 10,
        *,
        thread_id: str | None = None,
    ) -> list["ReflectionEvent"]:
        from .memory import ReflectionEvent

        clauses = ["user_id = ?"]
        params: list[object] = [clean_user_id(user_id)]
        if thread_id:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM reflection_events
                WHERE {' AND '.join(clauses)}
                ORDER BY id DESC
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
        return [
            ReflectionEvent(
                id=row["id"],
                user_id=row["user_id"],
                thread_id=row["thread_id"],
                source_event_ids=[int(event_id) for event_id in row["source_event_ids"].split(",") if event_id],
                summary=row["summary"],
                memory_count=row["memory_count"],
                profile_fields=[field for field in row["profile_fields"].split(",") if field],
                created_at=row["created_at"],
            )
            for row in rows
        ]

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
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO routing_events (
                    user_id, thread_id, user_text,
                    should_retrieve, retrieve_reason,
                    should_learn, learn_reason, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_user_id(user_id),
                    thread_id,
                    redact_sensitive(user_text),
                    int(should_retrieve),
                    retrieve_reason,
                    int(should_learn),
                    learn_reason,
                    now_iso(),
                ),
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
        from .memory import RoutingEvent

        clauses = ["user_id = ?"]
        params: list[object] = [clean_user_id(user_id)]
        if thread_id:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if learn is not None:
            clauses.append("should_learn = ?")
            params.append(int(learn))
        if retrieve is not None:
            clauses.append("should_retrieve = ?")
            params.append(int(retrieve))
        if reason:
            clauses.append("(retrieve_reason = ? OR learn_reason = ?)")
            params.extend([reason, reason])
        if text_query:
            clauses.append("user_text LIKE ?")
            params.append(f"%{text_query}%")
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM routing_events
                WHERE {' AND '.join(clauses)}
                ORDER BY id DESC
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
        return [
            RoutingEvent(
                id=row["id"],
                user_id=row["user_id"],
                thread_id=row["thread_id"],
                user_text=row["user_text"],
                should_retrieve=bool(row["should_retrieve"]),
                retrieve_reason=row["retrieve_reason"],
                should_learn=bool(row["should_learn"]),
                learn_reason=row["learn_reason"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

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
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO retrieval_events (
                    user_id, thread_id, user_text, memory_count, memory_ids, memory_preview, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_user_id(user_id),
                    thread_id,
                    redact_sensitive(user_text),
                    memory_count,
                    ",".join(str(memory_id) for memory_id in memory_ids),
                    redact_sensitive(memory_preview),
                    now_iso(),
                ),
            )

    def recent_retrieval_events(
        self,
        user_id: str = "default",
        limit: int = 10,
        *,
        thread_id: str | None = None,
    ) -> list["RetrievalEvent"]:
        from .memory import RetrievalEvent

        clauses = ["user_id = ?"]
        params: list[object] = [clean_user_id(user_id)]
        if thread_id:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM retrieval_events
                WHERE {' AND '.join(clauses)}
                ORDER BY id DESC
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
        return [
            RetrievalEvent(
                id=row["id"],
                user_id=row["user_id"],
                thread_id=row["thread_id"],
                user_text=row["user_text"],
                memory_count=row["memory_count"],
                memory_ids=[int(part) for part in row["memory_ids"].split(",") if part],
                memory_preview=row["memory_preview"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def add_dedupe_event(
        self,
        *,
        user_id: str,
        thread_id: str | None,
        removed_count: int,
        removed_ids: list[int],
        kept_ids: list[int],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO dedupe_events (
                    user_id, thread_id, removed_count, removed_ids, kept_ids, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_user_id(user_id),
                    thread_id,
                    removed_count,
                    ",".join(str(memory_id) for memory_id in removed_ids),
                    ",".join(str(memory_id) for memory_id in kept_ids),
                    now_iso(),
                ),
            )

    def recent_dedupe_events(
        self,
        user_id: str = "default",
        limit: int = 10,
        *,
        thread_id: str | None = None,
    ) -> list["DedupeEvent"]:
        from .memory import DedupeEvent

        clauses = ["user_id = ?"]
        params: list[object] = [clean_user_id(user_id)]
        if thread_id is not None:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM dedupe_events
                WHERE {' AND '.join(clauses)}
                ORDER BY id DESC
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
        return [
            DedupeEvent(
                id=row["id"],
                user_id=row["user_id"],
                thread_id=row["thread_id"],
                removed_count=row["removed_count"],
                removed_ids=[int(part) for part in row["removed_ids"].split(",") if part],
                kept_ids=[int(part) for part in row["kept_ids"].split(",") if part],
                created_at=row["created_at"],
            )
            for row in rows
        ]

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
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO memory_evolution_events (
                    user_id, thread_id, action, candidate_category, candidate_content,
                    target_memory_id, result_memory_id, reason, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    clean_user_id(user_id),
                    thread_id,
                    action,
                    candidate_category,
                    redact_sensitive(candidate_content),
                    target_memory_id,
                    result_memory_id,
                    reason,
                    now_iso(),
                ),
            )

    def recent_memory_evolution_events(
        self,
        user_id: str = "default",
        limit: int = 10,
        *,
        thread_id: str | None = None,
        action: str | None = None,
        reason: str | None = None,
    ) -> list["MemoryEvolutionEvent"]:
        from .memory import MemoryEvolutionEvent

        clauses = ["user_id = ?"]
        params: list[object] = [clean_user_id(user_id)]
        if thread_id is not None:
            clauses.append("thread_id = ?")
            params.append(thread_id)
        if action is not None:
            clauses.append("action = ?")
            params.append(action)
        if reason is not None:
            clauses.append("reason = ?")
            params.append(reason)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM memory_evolution_events
                WHERE {' AND '.join(clauses)}
                ORDER BY id DESC
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()
        return [
            MemoryEvolutionEvent(
                id=row["id"],
                user_id=row["user_id"],
                thread_id=row["thread_id"],
                action=row["action"],
                candidate_category=row["candidate_category"],
                candidate_content=row["candidate_content"],
                target_memory_id=row["target_memory_id"],
                result_memory_id=row["result_memory_id"],
                reason=row["reason"],
                created_at=row["created_at"],
            )
            for row in rows
        ]


class SqliteTranscriptRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_thread_summary(self, thread_id: str, user_id: str = "default") -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT summary FROM thread_summaries WHERE user_id = ? AND thread_id = ?", (clean_user_id(user_id), thread_id)).fetchone()
        return row["summary"] if row else None

    def update_thread_summary(self, thread_id: str, summary: str, user_id: str = "default") -> None:
        with self._connect() as conn:
            conn.execute("INSERT INTO thread_summaries (user_id, thread_id, summary, updated_at) VALUES (?, ?, ?, ?) ON CONFLICT(user_id, thread_id) DO UPDATE SET summary = excluded.summary, updated_at = excluded.updated_at", (clean_user_id(user_id), thread_id, redact_sensitive(summary), now_iso()))

    def add_message(self, thread_id: str, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (thread_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (thread_id, role, redact_sensitive(content), now_iso()),
            )

    def recent_messages(self, thread_id: str, limit: int) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content FROM messages
                WHERE thread_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (thread_id, limit),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]

    def thread_messages(self, thread_id: str, limit: int = 50) -> list["ThreadMessage"]:
        from .memory import ThreadMessage

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, thread_id, role, content, created_at
                FROM messages
                WHERE thread_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (thread_id, limit),
            ).fetchall()
        return [
            ThreadMessage(
                id=row["id"],
                thread_id=row["thread_id"],
                role=row["role"],
                content=row["content"],
                created_at=row["created_at"],
            )
            for row in rows
        ]
