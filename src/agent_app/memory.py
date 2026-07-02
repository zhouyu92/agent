from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class MemoryItem:
    id: int
    category: str
    content: str
    importance: int
    source: str
    created_at: str


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


class MemoryStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
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
                INSERT OR IGNORE INTO agent_profile (
                    id, identity, style_notes, boundaries, updated_at
                )
                VALUES (1, ?, ?, ?, ?)
                """,
                (
                    "一个像真人一样交流、会复盘但不伪装成人类的 agent。",
                    "温暖、坦诚、先理解再回答；必要时主动承认不确定。",
                    "不保存敏感凭据；不把猜测当事实；重要行动前说明影响。",
                    _now(),
                ),
            )
            self._ensure_column(conn, "memories", "user_id", "TEXT NOT NULL DEFAULT 'default'")

    def _ensure_column(self, conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def add_memory(
        self,
        category: str,
        content: str,
        importance: int,
        source: str,
        user_id: str = "default",
    ) -> bool:
        content = content.strip()
        if not content or _looks_sensitive(content):
            return False
        with self._connect() as conn:
            if self._has_similar_memory(conn, _clean_user_id(user_id), category.strip() or "general", content):
                return False
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO memories (user_id, category, content, importance, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (_clean_user_id(user_id), category.strip() or "general", content, max(1, min(5, importance)), source, _now()),
            )
        return cursor.rowcount > 0

    def _has_similar_memory(self, conn: sqlite3.Connection, user_id: str, category: str, content: str) -> bool:
        rows = conn.execute(
            """
            SELECT content FROM memories
            WHERE user_id = ? AND category = ?
            """,
            (user_id, category),
        ).fetchall()
        new_terms = _memory_terms(content)
        if len(new_terms) < 3:
            return False
        for row in rows:
            existing_terms = _memory_terms(row["content"])
            if len(existing_terms) < 3:
                continue
            overlap = len(new_terms & existing_terms)
            ratio = overlap / min(len(new_terms), len(existing_terms))
            if overlap >= 3 and ratio >= 0.65:
                return True
        return False

    def search_memories(self, query: str, limit: int = 5, user_id: str = "default") -> list[MemoryItem]:
        query_terms = _tokens(query)
        if not query_terms:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE user_id = ?",
                (_clean_user_id(user_id),),
            ).fetchall()

        scored: list[tuple[int, MemoryItem]] = []
        for row in rows:
            content_terms = _tokens(row["content"] + " " + row["category"] + " " + _category_alias(row["category"]))
            overlap = len(query_terms & content_terms)
            if overlap == 0:
                continue
            item = MemoryItem(
                id=row["id"],
                category=row["category"],
                content=row["content"],
                importance=row["importance"],
                source=row["source"],
                created_at=row["created_at"],
            )
            scored.append((overlap * 10 + item.importance, item))

        scored.sort(key=lambda pair: (-pair[0], -pair[1].importance, pair[1].id))
        return [item for _, item in scored[:limit]]

    def recent_memories(self, limit: int = 10, user_id: str = "default") -> list[MemoryItem]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM memories
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (_clean_user_id(user_id), limit),
            ).fetchall()
        return [
            MemoryItem(
                id=row["id"],
                category=row["category"],
                content=row["content"],
                importance=row["importance"],
                source=row["source"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def delete_memory(self, memory_id: int, user_id: str = "default") -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                DELETE FROM memories
                WHERE id = ? AND user_id = ?
                """,
                (memory_id, _clean_user_id(user_id)),
            )
        return cursor.rowcount > 0

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
                    _clean_user_id(user_id),
                    thread_id,
                    _redact_sensitive(user_text),
                    _redact_sensitive(assistant_text),
                    memory_count,
                    ",".join(profile_fields),
                    _now(),
                ),
            )

    def recent_learning_events(self, user_id: str = "default", limit: int = 10) -> list[LearningEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM learning_events
                WHERE user_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (_clean_user_id(user_id), limit),
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

    def add_message(self, thread_id: str, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (thread_id, role, content, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (thread_id, role, _redact_sensitive(content), _now()),
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

    def get_profile(self) -> AgentProfile:
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
                    _now(),
                ),
            )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean_user_id(user_id: str) -> str:
    cleaned = user_id.strip()
    return cleaned or "default"


def _tokens(text: str) -> set[str]:
    lowered = text.lower()
    words = set(re.findall(r"[a-z0-9_]{2,}", lowered))
    cjk = set(re.findall(r"[\u4e00-\u9fff]{2,}", lowered))
    chars = {char for char in lowered if "\u4e00" <= char <= "\u9fff"}
    return words | cjk | chars


def _memory_terms(text: str) -> set[str]:
    lowered = text.lower()
    terms = set(re.findall(r"[a-z0-9_]{2,}", lowered))
    terms |= set(re.findall(r"[\u4e00-\u9fff]{2,}", lowered))
    keywords = [
        "结论",
        "原因",
        "偏好",
        "回答",
        "结构",
        "直接",
        "详细",
        "推演",
        "温度",
        "风格",
        "边界",
        "原则",
    ]
    terms |= {keyword for keyword in keywords if keyword in text}
    return terms


def _category_alias(category: str) -> str:
    aliases = {
        "preference": "偏好 喜好",
        "fact": "事实 信息",
        "relationship": "关系",
        "feedback": "反馈 评价",
        "principle": "原则 规则",
        "general": "通用",
    }
    return aliases.get(category.lower(), "")


def _looks_sensitive(text: str) -> bool:
    patterns = [
        r"\bsk-[A-Za-z0-9_-]{16,}\b",
        r"(?i)\b(api[_ -]?key|token|secret|password|passwd)\b\s*[:=是]\s*\S+",
        r"(?i)\b(access[_ -]?key[_ -]?secret|access[_ -]?token)\b",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _redact_sensitive(text: str) -> str:
    redacted = re.sub(r"\bsk-[A-Za-z0-9_-]{16,}\b", "[REDACTED_SECRET]", text)
    redacted = re.sub(
        r"(?i)\b(api[_ -]?key|token|secret|password|passwd)\b(\s*[:=是]\s*)\S+",
        r"\1\2[REDACTED_SECRET]",
        redacted,
    )
    return redacted
