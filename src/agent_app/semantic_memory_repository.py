from __future__ import annotations

import sqlite3
from pathlib import Path

from .storage_utils import clean_user_id


class SqliteSemanticMemoryRepository:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def existing_contents(self, user_id: str, category: str) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT content FROM memories
                WHERE user_id = ? AND category = ?
                """,
                (clean_user_id(user_id), category),
            ).fetchall()
        return [row["content"] for row in rows]

    def insert_memory(
        self,
        *,
        user_id: str,
        category: str,
        content: str,
        importance: int,
        source: str,
        created_at: str,
    ) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO memories (
                    user_id, category, content, importance, source, created_at, last_confirmed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (clean_user_id(user_id), category, content, importance, source, created_at, created_at),
            )
        return cursor.rowcount > 0

    def insert_revision_memory(
        self,
        *,
        user_id: str,
        category: str,
        content: str,
        importance: int,
        source: str,
        created_at: str,
        supersedes_memory_id: int | None = None,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO memories (
                    user_id, category, content, importance, source, created_at,
                    status, supersedes_memory_id, reinforcement_count, last_reinforced_at, last_confirmed_at
                )
                VALUES (?, ?, ?, ?, ?, ?, 'active', ?, 0, NULL, ?)
                """,
                (
                    clean_user_id(user_id),
                    category,
                    content,
                    importance,
                    source,
                    created_at,
                    supersedes_memory_id,
                    created_at,
                ),
            )
        return int(cursor.lastrowid)

    def list_memories(self, user_id: str) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                "SELECT * FROM memories WHERE user_id = ?",
                (clean_user_id(user_id),),
            ).fetchall()

    def active_memories(self, user_id: str) -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ? AND status = 'active'
                """,
                (clean_user_id(user_id),),
            ).fetchall()

    def active_memories_by_ids(self, user_id: str, memory_ids: list[int]) -> list[sqlite3.Row]:
        if not memory_ids:
            return []
        placeholders = ",".join("?" for _ in memory_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM memories
                WHERE user_id = ? AND status = 'active' AND id IN ({placeholders})
                """,
                (clean_user_id(user_id), *memory_ids),
            ).fetchall()
        rows_by_id = {row["id"]: row for row in rows}
        return [rows_by_id[memory_id] for memory_id in memory_ids if memory_id in rows_by_id]

    def active_memory_by_id(self, user_id: str, memory_id: int) -> sqlite3.Row | None:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ? AND status = 'active' AND id = ?
                """,
                (clean_user_id(user_id), memory_id),
            ).fetchone()

    def recent_memories(self, user_id: str, limit: int, *, status: str = "active") -> list[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ? AND status = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (clean_user_id(user_id), status, limit),
            ).fetchall()

    def mark_memory_superseded(self, memory_id: int, user_id: str = "default") -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE memories
                SET status = 'superseded'
                WHERE id = ? AND user_id = ?
                """,
                (memory_id, clean_user_id(user_id)),
            )

    def archive_memory(self, memory_id: int, user_id: str = "default") -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE memories
                SET status = 'archived'
                WHERE id = ? AND user_id = ? AND status = 'active'
                """,
                (memory_id, clean_user_id(user_id)),
            )
        return cursor.rowcount > 0

    def reinforce_memory(self, memory_id: int, *, user_id: str = "default", created_at: str, increase_importance: bool = True) -> None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT importance, reinforcement_count
                FROM memories
                WHERE id = ? AND user_id = ?
                """,
                (memory_id, clean_user_id(user_id)),
            ).fetchone()
            if row is None:
                return
            importance = row["importance"]
            if increase_importance:
                importance = min(5, importance + 1)
            conn.execute(
                """
                UPDATE memories
                SET importance = ?, reinforcement_count = reinforcement_count + 1,
                    last_reinforced_at = ?, last_confirmed_at = ?
                WHERE id = ? AND user_id = ?
                """,
                (importance, created_at, created_at, memory_id, clean_user_id(user_id)),
            )

    def delete_memory(self, memory_id: int, user_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM memories
                WHERE id = ? AND user_id = ?
                """,
                (memory_id, clean_user_id(user_id)),
            )
        return cursor.rowcount > 0

    def delete_memories(self, memory_ids: list[int], user_id: str) -> int:
        if not memory_ids:
            return 0
        placeholders = ",".join("?" for _ in memory_ids)
        params = [*memory_ids, clean_user_id(user_id)]
        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                DELETE FROM memories
                WHERE id IN ({placeholders}) AND user_id = ?
                """,
                params,
            )
        return cursor.rowcount
