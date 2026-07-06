from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .semantic_memory import is_similar_memory, memory_terms, query_terms, score_memory_match
from .semantic_memory_repository import SqliteSemanticMemoryRepository
from .storage_utils import looks_sensitive, now_iso

if TYPE_CHECKING:
    from .memory import DedupeResult, MemoryEvolutionResult, MemoryItem


class SqliteSemanticMemoryStore:
    def __init__(self, db_path: str | Path | object) -> None:
        resolved_path = getattr(db_path, "db_path", db_path)
        self.repository = SqliteSemanticMemoryRepository(resolved_path)

    def search_memories(self, query: str, limit: int = 5, user_id: str = "default") -> list["MemoryItem"]:
        from .memory import MemoryItem

        if not query_terms(query):
            return []

        rows = self.repository.active_memories(user_id)
        scored: list[tuple[int, MemoryItem]] = []
        for row in rows:
            item = MemoryItem(
                id=row["id"],
                category=row["category"],
                content=row["content"],
                importance=row["importance"],
                source=row["source"],
                created_at=row["created_at"],
                status=row["status"],
            )
            score = score_memory_match(query, row["content"], row["category"], item.importance)
            if score == 0:
                continue
            scored.append((score, item))

        scored.sort(key=lambda pair: (-pair[0], -pair[1].importance, pair[1].id))
        return [item for _, item in scored[:limit]]

    def add_memory(
        self,
        category: str,
        content: str,
        importance: int,
        source: str,
        user_id: str = "default",
    ) -> bool:
        content = content.strip()
        if not content or looks_sensitive(content):
            return False
        normalized_category = category.strip() or "general"
        if is_similar_memory(self.repository.existing_contents(user_id, normalized_category), content):
            return False
        return self.repository.insert_memory(
            user_id=user_id,
            category=normalized_category,
            content=content,
            importance=max(1, min(5, importance)),
            source=source,
            created_at=now_iso(),
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
    ) -> "MemoryEvolutionResult":
        from .memory import MemoryEvolutionResult

        normalized_content = content.strip()
        normalized_category = category.strip() or "general"
        if not normalized_content or looks_sensitive(normalized_content) or len(normalized_content) < 4:
            return MemoryEvolutionResult(
                action="ignore",
                candidate_category=normalized_category,
                candidate_content=normalized_content,
                target_memory_id=None,
                result_memory_id=None,
                reason="low_value",
            )

        matched_row = self._best_active_match(user_id, normalized_category, normalized_content)
        if matched_row is None:
            memory_id = self.repository.insert_revision_memory(
                user_id=user_id,
                category=normalized_category,
                content=normalized_content,
                importance=max(1, min(5, importance)),
                source=source,
                created_at=now_iso(),
                supersedes_memory_id=None,
            )
            return MemoryEvolutionResult(
                action="add",
                candidate_category=normalized_category,
                candidate_content=normalized_content,
                target_memory_id=None,
                result_memory_id=memory_id,
                reason="new_memory",
            )

        if self._looks_like_correction(normalized_content):
            self.repository.mark_memory_superseded(matched_row["id"], user_id=user_id)
            memory_id = self.repository.insert_revision_memory(
                user_id=user_id,
                category=normalized_category,
                content=normalized_content,
                importance=max(1, min(5, importance)),
                source=source,
                created_at=now_iso(),
                supersedes_memory_id=matched_row["id"],
            )
            return MemoryEvolutionResult(
                action="revise",
                candidate_category=normalized_category,
                candidate_content=normalized_content,
                target_memory_id=matched_row["id"],
                result_memory_id=memory_id,
                reason="correction_phrase",
            )

        self.repository.reinforce_memory(matched_row["id"], user_id=user_id, created_at=now_iso())
        return MemoryEvolutionResult(
            action="reinforce",
            candidate_category=normalized_category,
            candidate_content=normalized_content,
            target_memory_id=matched_row["id"],
            result_memory_id=matched_row["id"],
            reason="confirmed_existing_memory",
        )

    def recent_memories(self, limit: int = 10, user_id: str = "default") -> list["MemoryItem"]:
        from .memory import MemoryItem

        rows = self.repository.recent_memories(user_id, limit)
        return [
            MemoryItem(
                id=row["id"],
                category=row["category"],
                content=row["content"],
                importance=row["importance"],
                source=row["source"],
                created_at=row["created_at"],
                status=row["status"],
            )
            for row in rows
        ]

    def delete_memory(self, memory_id: int, user_id: str = "default") -> bool:
        return self.repository.delete_memory(memory_id, user_id)

    def dedupe_memories(self, user_id: str = "default") -> "DedupeResult":
        from .memory import DedupeResult

        rows = self.repository.list_memories(user_id)
        rows_by_category: dict[str, list[object]] = {}
        for row in rows:
            rows_by_category.setdefault(row["category"], []).append(row)

        delete_ids: list[int] = []
        kept_ids: list[int] = []
        for category_rows in rows_by_category.values():
            kept_rows: list[tuple[int, str]] = []
            for row in sorted(category_rows, key=lambda item: (-item["importance"], item["id"])):
                content = row["content"]
                matched_kept_id: int | None = None
                for kept_id, kept_content in kept_rows:
                    if is_similar_memory([kept_content], content):
                        matched_kept_id = kept_id
                        break
                if matched_kept_id is not None:
                    delete_ids.append(row["id"])
                    if matched_kept_id not in kept_ids:
                        kept_ids.append(matched_kept_id)
                    continue
                kept_rows.append((row["id"], content))

        removed_count = self.repository.delete_memories(delete_ids, user_id)
        if removed_count != len(delete_ids):
            delete_ids = delete_ids[:removed_count]
        return DedupeResult(removed_count=removed_count, removed_ids=delete_ids, kept_ids=kept_ids)

    def _best_active_match(self, user_id: str, category: str, content: str):
        candidate_terms = memory_terms(content)
        for row in self.repository.active_memories(user_id):
            if row["category"] != category:
                continue
            if is_similar_memory([row["content"]], content):
                return row
            overlap = len(candidate_terms & memory_terms(row["content"]))
            if overlap >= 1:
                return row
        return None

    @staticmethod
    def _looks_like_correction(text: str) -> bool:
        markers = ("不是", "而是", "改成", "更准确地说", "实际上", "以后以", "以后回答")
        return any(marker in text for marker in markers)
