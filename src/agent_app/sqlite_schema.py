from __future__ import annotations

import sqlite3
from pathlib import Path

from .storage_utils import now_iso


def ensure_sqlite_schema(db_path: str | Path) -> None:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL DEFAULT 'default',
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                importance INTEGER NOT NULL DEFAULT 3,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(user_id, content)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_profile (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                identity TEXT NOT NULL,
                style_notes TEXT NOT NULL,
                boundaries TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS learning_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                user_text TEXT NOT NULL,
                assistant_text TEXT NOT NULL,
                memory_count INTEGER NOT NULL,
                profile_fields TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS routing_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                user_text TEXT NOT NULL,
                should_retrieve INTEGER NOT NULL,
                retrieve_reason TEXT NOT NULL,
                should_learn INTEGER NOT NULL,
                learn_reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS retrieval_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                user_text TEXT NOT NULL,
                memory_count INTEGER NOT NULL,
                memory_ids TEXT NOT NULL,
                memory_preview TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dedupe_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                thread_id TEXT,
                removed_count INTEGER NOT NULL,
                removed_ids TEXT NOT NULL,
                kept_ids TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_evolution_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                thread_id TEXT,
                action TEXT NOT NULL,
                candidate_category TEXT NOT NULL,
                candidate_content TEXT NOT NULL,
                target_memory_id INTEGER,
                result_memory_id INTEGER,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT OR IGNORE INTO agent_profile (
                id, identity, style_notes, boundaries, updated_at
            )
            VALUES (1, ?, ?, ?, ?)
            """,
            (
                "一个像真人一样交流、会复盘但不伪装成人类的 agent。",
                "温暖、坦诚、先理解再回答；必要时主动承认不确定。",
                "不保存敏感凭据；不把猜测当事实；重要行动前说明影响。",
                now_iso(),
            ),
        )
        _ensure_column(conn, "memories", "user_id", "TEXT NOT NULL DEFAULT 'default'")
        _ensure_column(conn, "memories", "status", "TEXT NOT NULL DEFAULT 'active'")
        _ensure_column(conn, "memories", "supersedes_memory_id", "INTEGER")
        _ensure_column(conn, "memories", "reinforcement_count", "INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "memories", "last_reinforced_at", "TEXT")
        _ensure_column(conn, "retrieval_events", "memory_ids", "TEXT NOT NULL DEFAULT ''")
        _ensure_column(conn, "dedupe_events", "thread_id", "TEXT")
        _ensure_column(conn, "dedupe_events", "kept_ids", "TEXT NOT NULL DEFAULT ''")
        conn.commit()
    finally:
        conn.close()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
