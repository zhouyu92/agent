from agent_app.memory import MemoryStore
from agent_app.vector_smoke import VectorSmokeResult, run_vector_smoke


class FakeIndexer:
    def __init__(self):
        self.indexed = []
        self.removed = []

    def index_memory(self, memory, *, user_id):
        self.indexed.append((memory, user_id))
        return True

    def remove_memory(self, *, memory_id):
        self.removed.append(memory_id)


class FakeSearcher:
    def __init__(self):
        self.calls = []
        self.memory_ids = []

    def search_memory_ids(self, query, *, user_id, limit):
        self.calls.append((query, user_id, limit))
        return self.memory_ids


def test_run_vector_smoke_indexes_searches_and_cleans_up(tmp_path):
    memory = MemoryStore(tmp_path / "agent.db")
    indexer = FakeIndexer()
    searcher = FakeSearcher()

    def after_index(memory_id):
        searcher.memory_ids = [memory_id]

    result = run_vector_smoke(
        memory,
        indexer=indexer,
        searcher=searcher,
        user_id="alice",
        after_index=after_index,
    )

    assert result.ok is True
    assert result.memory_id is not None
    assert result.found_ids == [result.memory_id]
    assert len(indexer.indexed) == 1
    assert indexer.indexed[0][0].id == result.memory_id
    assert indexer.indexed[0][1] == "alice"
    assert searcher.calls == [("vector smoke unique memory", "alice", 5)]
    assert indexer.removed == [result.memory_id]
    assert memory.search_memories("vector smoke unique memory", user_id="alice") == []


def test_run_vector_smoke_reports_search_miss_and_still_cleans_up(tmp_path):
    memory = MemoryStore(tmp_path / "agent.db")
    indexer = FakeIndexer()
    searcher = FakeSearcher()

    result = run_vector_smoke(
        memory,
        indexer=indexer,
        searcher=searcher,
        user_id="alice",
    )

    assert result == VectorSmokeResult(ok=False, memory_id=result.memory_id, found_ids=[])
    assert result.memory_id is not None
    assert indexer.removed == [result.memory_id]
    assert memory.search_memories("vector smoke unique memory", user_id="alice") == []
