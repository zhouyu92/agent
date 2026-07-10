from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from .config import AgentConfig
from .memory import MemoryItem, MemoryStore
from .vector_memory import VectorMemoryIndexer, VectorMemorySearcher


class SmokeIndexer(Protocol):
    def index_memory(self, memory: MemoryItem, *, user_id: str) -> bool:
        ...

    def remove_memory(self, *, memory_id: int) -> None:
        ...


class SmokeSearcher(Protocol):
    def search_memory_ids(self, query: str, *, user_id: str, limit: int) -> list[int]:
        ...


@dataclass(frozen=True)
class VectorSmokeResult:
    ok: bool
    memory_id: int | None
    found_ids: list[int]


SMOKE_CONTENT = "vector smoke unique memory"


def run_vector_smoke(
    memory: MemoryStore,
    *,
    indexer: SmokeIndexer,
    searcher: SmokeSearcher,
    user_id: str,
    after_index: Callable[[int], None] | None = None,
) -> VectorSmokeResult:
    memory_id: int | None = None
    try:
        memory.add_memory(
            category="fact",
            content=SMOKE_CONTENT,
            importance=1,
            source="vector-smoke",
            user_id=user_id,
        )
        memory_item = memory.recent_memories(user_id=user_id, limit=1)[0]
        memory_id = memory_item.id
        indexer.index_memory(memory_item, user_id=user_id)
        if after_index is not None:
            after_index(memory_id)
        found_ids = searcher.search_memory_ids(SMOKE_CONTENT, user_id=user_id, limit=5)
        return VectorSmokeResult(ok=memory_id in found_ids, memory_id=memory_id, found_ids=found_ids)
    finally:
        if memory_id is not None:
            try:
                indexer.remove_memory(memory_id=memory_id)
            finally:
                memory.delete_memory(memory_id, user_id=user_id)


def main() -> None:
    config = AgentConfig.from_env()
    memory = MemoryStore(config.memory_db_path)
    indexer = VectorMemoryIndexer(config)
    searcher = VectorMemorySearcher(config)
    result = run_vector_smoke(memory, indexer=indexer, searcher=searcher, user_id=config.user_id)
    print(f"ok: {result.ok}")
    print(f"memory_id: {result.memory_id}")
    print(f"found_ids: {','.join(str(memory_id) for memory_id in result.found_ids)}")
    raise SystemExit(0 if result.ok else 1)


if __name__ == "__main__":
    main()
