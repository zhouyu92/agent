from agent_app.semantic_memory_repository import SqliteSemanticMemoryRepository
from agent_app.sqlite_schema import ensure_sqlite_schema


def test_sqlite_semantic_memory_repository_stores_lists_and_deletes_rows(tmp_path):
    db_path = tmp_path / "agent.db"
    ensure_sqlite_schema(db_path)
    repo = SqliteSemanticMemoryRepository(db_path)

    saved = repo.insert_memory(
        user_id="alice",
        category="preference",
        content="用户喜欢先给结论。",
        importance=4,
        source="conversation",
        created_at="2026-07-03T00:00:00+00:00",
    )

    memories = repo.list_memories("alice")
    deleted = repo.delete_memory(memories[0]["id"], "alice")

    assert saved is True
    assert len(memories) == 1
    assert memories[0]["content"] == "用户喜欢先给结论。"
    assert deleted is True
    assert repo.list_memories("alice") == []
