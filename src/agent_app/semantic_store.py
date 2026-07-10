from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .memory_evolution import (
    choose_evolution_action,
    has_contextual_preference_scope,
    looks_like_fact_correction,
    looks_like_weak_signal,
)
from .semantic_memory import is_similar_memory, memory_terms, query_terms, score_memory_match
from .semantic_memory_repository import SqliteSemanticMemoryRepository
from .sqlite_schema import ensure_sqlite_schema
from .storage_utils import looks_sensitive, now_iso

if TYPE_CHECKING:
    from .memory import DedupeResult, MemoryEvolutionResult, MemoryItem


class VectorMemoryIndexer(Protocol):
    def index_memory(self, memory: "MemoryItem", *, user_id: str) -> bool:
        ...


class VectorMemorySearcher(Protocol):
    def search_memory_ids(self, query: str, *, user_id: str, limit: int) -> list[int]:
        ...


class SqliteSemanticMemoryStore:
    def __init__(
        self,
        db_path: str | Path | object,
        vector_indexer: VectorMemoryIndexer | None = None,
        vector_searcher: VectorMemorySearcher | None = None,
    ) -> None:
        resolved_path = getattr(db_path, "db_path", db_path)
        ensure_sqlite_schema(resolved_path)
        self.repository = SqliteSemanticMemoryRepository(resolved_path)
        self.vector_indexer = vector_indexer
        self.vector_searcher = vector_searcher

    def search_memories(self, query: str, limit: int = 5, user_id: str = "default", status: str = "active") -> list["MemoryItem"]:
        from .memory import MemoryItem

        if not query_terms(query):
            return []

        vector_results = self._search_memories_with_vector(query, limit=limit, user_id=user_id) if status == "active" else []
        if vector_results:
            return vector_results

        rows = self.repository.active_memories(user_id) if status == "active" else self.repository.recent_memories(user_id, 200, status=status)
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
                last_confirmed_at=row["last_confirmed_at"],
            )
            score = score_memory_match(query, row["content"], row["category"], item.importance)
            if score == 0:
                continue
            scored.append((score, item))

        scored.sort(key=lambda pair: (-pair[0], -pair[1].importance, pair[1].id))
        return [item for _, item in scored[:limit]]

    def _search_memories_with_vector(self, query: str, *, limit: int, user_id: str) -> list["MemoryItem"]:
        from .memory import MemoryItem

        if self.vector_searcher is None:
            return []
        try:
            memory_ids = self.vector_searcher.search_memory_ids(query, user_id=user_id, limit=limit)
        except Exception:
            return []
        rows = self.repository.active_memories_by_ids(user_id, memory_ids)
        return [
            MemoryItem(
                id=row["id"],
                category=row["category"],
                content=row["content"],
                importance=row["importance"],
                source=row["source"],
                created_at=row["created_at"],
                status=row["status"],
                last_confirmed_at=row["last_confirmed_at"],
            )
            for row in rows
        ]

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
        if (
            not normalized_content
            or looks_sensitive(normalized_content)
            or len(normalized_content) < 4
            or looks_like_weak_signal(normalized_content)
        ):
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
            created_at = now_iso()
            memory_id = self.repository.insert_revision_memory(
                user_id=user_id,
                category=normalized_category,
                content=normalized_content,
                importance=max(1, min(5, importance)),
                source=source,
                created_at=created_at,
                supersedes_memory_id=None,
            )
            self._index_memory_if_configured(
                memory_id=memory_id,
                category=normalized_category,
                content=normalized_content,
                importance=importance,
                source=source,
                created_at=created_at,
                user_id=user_id,
            )
            return MemoryEvolutionResult(
                action="add",
                candidate_category=normalized_category,
                candidate_content=normalized_content,
                target_memory_id=None,
                result_memory_id=memory_id,
                reason="new_memory",
            )

        action, reason = choose_evolution_action(normalized_content, matched_row["content"])
        if action == "ignore":
            return MemoryEvolutionResult(
                action="ignore",
                candidate_category=normalized_category,
                candidate_content=normalized_content,
                target_memory_id=matched_row["id"],
                result_memory_id=None,
                reason=reason,
            )
        if action == "revise":
            self.repository.mark_memory_superseded(matched_row["id"], user_id=user_id)
            self._remove_vector_if_configured(memory_id=matched_row["id"])
            created_at = now_iso()
            memory_id = self.repository.insert_revision_memory(
                user_id=user_id,
                category=normalized_category,
                content=normalized_content,
                importance=max(1, min(5, importance)),
                source=source,
                created_at=created_at,
                supersedes_memory_id=matched_row["id"],
            )
            self._index_memory_if_configured(
                memory_id=memory_id,
                category=normalized_category,
                content=normalized_content,
                importance=importance,
                source=source,
                created_at=created_at,
                user_id=user_id,
            )
            return MemoryEvolutionResult(
                action="revise",
                candidate_category=normalized_category,
                candidate_content=normalized_content,
                target_memory_id=matched_row["id"],
                result_memory_id=memory_id,
                reason=reason,
            )

        self.repository.reinforce_memory(matched_row["id"], user_id=user_id, created_at=now_iso())
        return MemoryEvolutionResult(
            action="reinforce",
            candidate_category=normalized_category,
            candidate_content=normalized_content,
            target_memory_id=matched_row["id"],
            result_memory_id=matched_row["id"],
            reason=reason,
        )

    def recent_memories(self, limit: int = 10, user_id: str = "default", status: str = "active") -> list["MemoryItem"]:
        from .memory import MemoryItem

        rows = self.repository.recent_memories(user_id, limit, status=status)
        return [
            MemoryItem(
                id=row["id"],
                category=row["category"],
                content=row["content"],
                importance=row["importance"],
                source=row["source"],
                created_at=row["created_at"],
                status=row["status"],
                last_confirmed_at=row["last_confirmed_at"],
            )
            for row in rows
        ]

    def delete_memory(self, memory_id: int, user_id: str = "default") -> bool:
        deleted = self.repository.delete_memory(memory_id, user_id)
        if deleted:
            self._remove_vector_if_configured(memory_id=memory_id)
        return deleted

    def archive_memory(self, memory_id: int, user_id: str = "default") -> bool:
        archived = self.repository.archive_memory(memory_id, user_id)
        if archived:
            self._remove_vector_if_configured(memory_id=memory_id)
        return archived

    def confirm_memory(self, memory_id: int, user_id: str = "default") -> bool:
        row = self.repository.active_memory_by_id(user_id, memory_id)
        if row is None:
            return False
        self.repository.reinforce_memory(memory_id, user_id=user_id, created_at=now_iso(), increase_importance=False)
        return True

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
        for memory_id in delete_ids:
            self._remove_vector_if_configured(memory_id=memory_id)
        return DedupeResult(removed_count=removed_count, removed_ids=delete_ids, kept_ids=kept_ids)

    def _best_active_match(self, user_id: str, category: str, content: str):
        vector_match = self._best_vector_active_match(user_id, category, content)
        if vector_match is not None:
            return vector_match

        candidate_terms = memory_terms(content)
        candidate_is_contextual_preference = category == "preference" and has_contextual_preference_scope(content)
        for row in self.repository.active_memories(user_id):
            if row["category"] != category:
                continue
            row_is_contextual_preference = category == "preference" and has_contextual_preference_scope(row["content"])
            if candidate_is_contextual_preference and not row_is_contextual_preference:
                continue
            if is_similar_memory([row["content"]], content):
                return row
            if looks_like_fact_correction(content, row["content"]):
                return row
            overlap = len(candidate_terms & memory_terms(row["content"]))
            if overlap >= 1:
                return row
        return None

    def _best_vector_active_match(self, user_id: str, category: str, content: str):
        if self.vector_searcher is None:
            return None
        try:
            memory_ids = self.vector_searcher.search_memory_ids(content, user_id=user_id, limit=5)
        except Exception:
            return None
        for row in self.repository.active_memories_by_ids(user_id, memory_ids):
            if row["category"] == category:
                return row
        return None

    def _index_memory_if_configured(
        self,
        *,
        memory_id: int,
        category: str,
        content: str,
        importance: int,
        source: str,
        created_at: str,
        user_id: str,
    ) -> None:
        if self.vector_indexer is None:
            return

        from .memory import MemoryItem

        item = MemoryItem(
            id=memory_id,
            category=category,
            content=content,
            importance=max(1, min(5, importance)),
            source=source,
            created_at=created_at,
            status="active",
            last_confirmed_at=created_at,
        )
        try:
            self.vector_indexer.index_memory(item, user_id=user_id)
        except Exception:
            return

    def _remove_vector_if_configured(self, *, memory_id: int) -> None:
        if self.vector_indexer is None:
            return
        remove_memory = getattr(self.vector_indexer, "remove_memory", None)
        if remove_memory is None:
            return
        try:
            remove_memory(memory_id=memory_id)
        except Exception:
            return
