from agent_app.sqlite_records import SqliteAuditRepository, SqliteProfileRepository, SqliteTranscriptRepository
from agent_app.sqlite_schema import ensure_sqlite_schema


def test_sqlite_profile_repository_updates_and_reads_profile(tmp_path):
    db_path = tmp_path / "agent.db"
    ensure_sqlite_schema(db_path)
    repo = SqliteProfileRepository(db_path)

    before = repo.get_profile()
    repo.update_profile(style_notes="更坦诚。")
    after = repo.get_profile()

    assert "像真人一样" in before.identity
    assert after.style_notes == "更坦诚。"


def test_sqlite_audit_repository_writes_and_reads_events(tmp_path):
    db_path = tmp_path / "agent.db"
    ensure_sqlite_schema(db_path)
    repo = SqliteAuditRepository(db_path)

    repo.add_routing_event(
        user_id="alice",
        thread_id="t1",
        user_text="谢谢",
        should_retrieve=False,
        retrieve_reason="low_signal",
        should_learn=False,
        learn_reason="low_signal",
    )
    repo.add_retrieval_event(
        user_id="alice",
        thread_id="t1",
        user_text="你还记得吗？",
        memory_count=1,
        memory_ids=[3],
        memory_preview="用户喜欢先给结论再补充原因。",
    )
    repo.add_learning_event(
        user_id="alice",
        thread_id="t1",
        user_text="以后回答先给结论。",
        assistant_text="好。",
        memory_count=1,
        profile_fields=["style_notes"],
    )
    repo.add_reflection_event(
        user_id="alice",
        thread_id="t1",
        source_event_ids=[2, 3],
        summary="用户稳定偏好先给结论。",
        memory_count=1,
        profile_fields=["style_notes"],
    )

    assert repo.recent_routing_events(user_id="alice", limit=1)[0].thread_id == "t1"
    assert repo.recent_retrieval_events(user_id="alice", thread_id="t1", limit=1)[0].memory_ids == [3]
    assert repo.recent_learning_events(user_id="alice", limit=1)[0].profile_fields == ["style_notes"]
    reflection = repo.recent_reflection_events(user_id="alice", limit=1, thread_id="t1")[0]
    assert reflection.source_event_ids == [2, 3]
    assert reflection.summary == "用户稳定偏好先给结论。"
    assert reflection.memory_count == 1


def test_sqlite_transcript_repository_writes_and_reads_messages(tmp_path):
    db_path = tmp_path / "agent.db"
    ensure_sqlite_schema(db_path)
    repo = SqliteTranscriptRepository(db_path)

    repo.add_message("t1", "user", "你好")
    repo.add_message("t1", "assistant", "你好，我在。")

    assert [(message.role, message.content) for message in repo.thread_messages("t1", limit=10)] == [
        ("user", "你好"),
        ("assistant", "你好，我在。"),
    ]
