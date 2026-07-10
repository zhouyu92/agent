from agent_app.memory import DedupeResult, MemoryStore
from agent_app.store import (
    AuditStore,
    MemoryProfileStore,
    SemanticMemoryStore,
    SqliteAuditStore,
    SqliteCliStore,
    SqliteLongTermStore,
    SqliteProfileStore,
    SqliteSemanticMemoryStore,
    SqliteTranscriptStore,
    ThreadTranscriptStore,
)


def test_sqlite_long_term_store_delegates_to_memory_store(tmp_path):
    memory = MemoryStore(tmp_path / "agent.db")
    store = SqliteLongTermStore(memory)

    saved = store.add_memory(
        category="preference",
        content="用户喜欢先给结论再补充原因。",
        importance=4,
        source="conversation",
        user_id="alice",
    )

    memories = store.search_memories("回答偏好", limit=5, user_id="alice")

    assert saved is True
    assert memories[0].content == "用户喜欢先给结论再补充原因。"


def test_sqlite_long_term_store_exposes_profile_and_audits(tmp_path):
    memory = MemoryStore(tmp_path / "agent.db")
    store = SqliteLongTermStore(memory)

    store.update_profile(style_notes="更坦诚。")
    store.add_routing_event(
        user_id="alice",
        thread_id="t1",
        user_text="谢谢",
        should_retrieve=False,
        retrieve_reason="low_signal",
        should_learn=False,
        learn_reason="low_signal",
    )
    store.add_retrieval_event(
        user_id="alice",
        thread_id="t1",
        user_text="你还记得吗？",
        memory_count=1,
        memory_ids=[3],
        memory_preview="用户喜欢先给结论再补充原因。",
    )
    store.add_learning_event(
        user_id="alice",
        thread_id="t1",
        user_text="以后回答先给结论。",
        assistant_text="好。",
        memory_count=1,
        profile_fields=["style_notes"],
    )
    store.add_dedupe_event(
        user_id="alice",
        thread_id="t1",
        removed_count=1,
        removed_ids=[3],
        kept_ids=[2],
    )

    profile = store.get_profile()
    routing = memory.recent_routing_events(user_id="alice", limit=1)[0]
    retrieval = memory.recent_retrieval_events(user_id="alice", thread_id="t1", limit=1)[0]
    learning = memory.recent_learning_events(user_id="alice", limit=1)[0]
    dedupe = memory.recent_dedupe_events(user_id="alice", limit=1, thread_id="t1")[0]

    assert profile.style_notes == "更坦诚。"
    assert routing.thread_id == "t1"
    assert retrieval.memory_ids == [3]
    assert learning.profile_fields == ["style_notes"]
    assert dedupe.thread_id == "t1"
    assert dedupe.removed_ids == [3]
    assert dedupe.kept_ids == [2]


def test_memory_evolution_event_round_trip(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")

    store.add_memory_evolution_event(
        user_id="alice",
        thread_id="t1",
        action="revise",
        candidate_category="preference",
        candidate_content="以后回答先给结论再补原因。",
        target_memory_id=1,
        result_memory_id=2,
        reason="correction_phrase",
    )
    events = store.recent_memory_evolution_events(user_id="alice", limit=1, thread_id="t1")

    assert len(events) == 1
    assert events[0].action == "revise"
    assert events[0].target_memory_id == 1
    assert events[0].result_memory_id == 2
    assert events[0].reason == "correction_phrase"


def test_sqlite_audit_store_round_trips_memory_evolution_events(tmp_path):
    memory = MemoryStore(tmp_path / "agent.db")
    store = SqliteAuditStore(memory)

    store.add_memory_evolution_event(
        user_id="alice",
        thread_id="t1",
        action="reinforce",
        candidate_category="preference",
        candidate_content="用户还是喜欢先给结论。",
        target_memory_id=3,
        result_memory_id=3,
        reason="confirmed_existing_memory",
    )
    event = store.recent_memory_evolution_events(user_id="alice", limit=1, thread_id="t1")[0]

    assert event.action == "reinforce"
    assert event.result_memory_id == 3


def test_sqlite_audit_store_filters_memory_evolution_events_by_reason(tmp_path):
    memory = MemoryStore(tmp_path / "agent.db")
    store = SqliteAuditStore(memory)

    store.add_memory_evolution_event(
        user_id="alice",
        thread_id="t1",
        action="ignore",
        candidate_category="preference",
        candidate_content="用户偏好回答时先给结论。",
        target_memory_id=3,
        result_memory_id=None,
        reason="no_new_information",
    )
    store.add_memory_evolution_event(
        user_id="alice",
        thread_id="t1",
        action="revise",
        candidate_category="preference",
        candidate_content="以后回答先给结论。",
        target_memory_id=3,
        result_memory_id=4,
        reason="correction_phrase",
    )

    events = store.recent_memory_evolution_events(
        user_id="alice",
        limit=10,
        thread_id="t1",
        reason="no_new_information",
    )

    assert len(events) == 1
    assert events[0].action == "ignore"
    assert events[0].reason == "no_new_information"


def test_sqlite_long_term_store_matches_split_store_protocols(tmp_path):
    memory = MemoryStore(tmp_path / "agent.db")
    store = SqliteLongTermStore(memory)

    assert isinstance(store, SemanticMemoryStore)
    assert isinstance(store, MemoryProfileStore)
    assert isinstance(store, AuditStore)


def test_sqlite_long_term_store_exposes_split_sqlite_adapters(tmp_path):
    memory = MemoryStore(tmp_path / "agent.db")
    store = SqliteLongTermStore(memory)

    assert isinstance(store.semantic_memory_store, SqliteSemanticMemoryStore)
    assert isinstance(store.profile_store, SqliteProfileStore)
    assert isinstance(store.audit_store, SqliteAuditStore)
    assert isinstance(store.transcript_store, SqliteTranscriptStore)
    assert isinstance(store.transcript_store, ThreadTranscriptStore)


def test_sqlite_profile_store_updates_profile_without_memory_store_method(tmp_path):
    memory = MemoryStore(tmp_path / "agent.db")
    store = SqliteProfileStore(memory)

    def fail_update_profile(**kwargs):
        raise AssertionError("should not call MemoryStore.update_profile")

    memory.update_profile = fail_update_profile  # type: ignore[method-assign]

    store.update_profile(style_notes="更坦诚。")
    reloaded = MemoryStore(tmp_path / "agent.db").get_profile()

    assert reloaded.style_notes == "更坦诚。"


def test_sqlite_audit_store_writes_events_without_memory_store_methods(tmp_path):
    memory = MemoryStore(tmp_path / "agent.db")
    store = SqliteAuditStore(memory)

    def fail_add_event(**kwargs):
        raise AssertionError("should not call MemoryStore audit methods")

    memory.add_routing_event = fail_add_event  # type: ignore[method-assign]
    memory.add_retrieval_event = fail_add_event  # type: ignore[method-assign]
    memory.add_learning_event = fail_add_event  # type: ignore[method-assign]
    memory.add_dedupe_event = fail_add_event  # type: ignore[method-assign]

    store.add_routing_event(
        user_id="alice",
        thread_id="t1",
        user_text="谢谢",
        should_retrieve=False,
        retrieve_reason="low_signal",
        should_learn=False,
        learn_reason="low_signal",
    )
    store.add_retrieval_event(
        user_id="alice",
        thread_id="t1",
        user_text="你还记得吗？",
        memory_count=1,
        memory_ids=[3],
        memory_preview="用户喜欢先给结论再补充原因。",
    )
    store.add_learning_event(
        user_id="alice",
        thread_id="t1",
        user_text="以后回答先给结论。",
        assistant_text="好。",
        memory_count=1,
        profile_fields=["style_notes"],
    )
    store.add_dedupe_event(user_id="alice", thread_id="t1", removed_count=1, removed_ids=[3], kept_ids=[2])

    reloaded = MemoryStore(tmp_path / "agent.db")
    assert reloaded.recent_routing_events(user_id="alice", limit=1)[0].thread_id == "t1"
    assert reloaded.recent_retrieval_events(user_id="alice", thread_id="t1", limit=1)[0].memory_ids == [3]
    assert reloaded.recent_learning_events(user_id="alice", limit=1)[0].profile_fields == ["style_notes"]
    event = reloaded.recent_dedupe_events(user_id="alice", limit=1, thread_id="t1")[0]
    assert event.removed_ids == [3]
    assert event.kept_ids == [2]


def test_sqlite_transcript_store_writes_messages_without_memory_store_method(tmp_path):
    memory = MemoryStore(tmp_path / "agent.db")
    store = SqliteTranscriptStore(memory)

    def fail_add_message(thread_id: str, role: str, content: str) -> None:
        raise AssertionError("should not call MemoryStore.add_message")

    memory.add_message = fail_add_message  # type: ignore[method-assign]

    store.add_message("t1", "user", "你好")
    store.add_message("t1", "assistant", "你好，我在。")

    reloaded = MemoryStore(tmp_path / "agent.db")
    assert [(message.role, message.content) for message in reloaded.thread_messages("t1", limit=10)] == [
        ("user", "你好"),
        ("assistant", "你好，我在。"),
    ]


def test_sqlite_cli_store_reads_profile_and_thread_audit_without_memory_methods(tmp_path):
    memory = MemoryStore(tmp_path / "agent.db")
    cli_store = SqliteCliStore(memory)
    profile_store = SqliteProfileStore(memory)
    audit_store = SqliteAuditStore(memory)
    transcript_store = SqliteTranscriptStore(memory)

    profile_store.update_profile(style_notes="更坦诚。")
    transcript_store.add_message("t1", "user", "你好")
    transcript_store.add_message("t1", "assistant", "你好，我在。")
    audit_store.add_routing_event(
        user_id="alice",
        thread_id="t1",
        user_text="你好",
        should_retrieve=False,
        retrieve_reason="low_signal",
        should_learn=False,
        learn_reason="low_signal",
    )
    audit_store.add_retrieval_event(
        user_id="alice",
        thread_id="t1",
        user_text="你还记得吗？",
        memory_count=1,
        memory_ids=[3],
        memory_preview="用户喜欢先给结论再补充原因。",
    )
    audit_store.add_learning_event(
        user_id="alice",
        thread_id="t1",
        user_text="以后回答先给结论。",
        assistant_text="好。",
        memory_count=1,
        profile_fields=["style_notes"],
    )
    audit_store.add_dedupe_event(user_id="alice", thread_id="t1", removed_count=1, removed_ids=[3], kept_ids=[2])

    def fail(*args, **kwargs):
        raise AssertionError("should not call MemoryStore query helpers")

    memory.get_profile = fail  # type: ignore[method-assign]
    memory.thread_messages = fail  # type: ignore[method-assign]
    memory.recent_routing_events = fail  # type: ignore[method-assign]
    memory.recent_retrieval_events = fail  # type: ignore[method-assign]
    memory.recent_learning_events = fail  # type: ignore[method-assign]
    memory.recent_dedupe_events = fail  # type: ignore[method-assign]

    assert cli_store.get_profile().style_notes == "更坦诚。"
    assert [(message.role, message.content) for message in cli_store.thread_messages("t1", limit=10)] == [
        ("user", "你好"),
        ("assistant", "你好，我在。"),
    ]
    assert cli_store.recent_routing_events(user_id="alice", limit=1)[0].thread_id == "t1"
    assert cli_store.recent_retrieval_events(user_id="alice", thread_id="t1", limit=1)[0].memory_ids == [3]
    assert cli_store.recent_learning_events(user_id="alice", limit=1)[0].profile_fields == ["style_notes"]
    event = cli_store.recent_dedupe_events(user_id="alice", limit=1, thread_id="t1")[0]
    assert event.removed_ids == [3]
    assert event.kept_ids == [2]


def test_sqlite_cli_store_delegates_reads_to_split_adapters(tmp_path):
    memory = MemoryStore(tmp_path / "agent.db")
    cli_store = SqliteCliStore(memory)
    calls = []

    def track_get_profile():
        calls.append("profile")
        return MemoryStore(tmp_path / "agent.db").get_profile()

    def track_thread_messages(thread_id: str, limit: int = 50):
        calls.append(("messages", thread_id, limit))
        return MemoryStore(tmp_path / "agent.db").thread_messages(thread_id, limit=limit)

    def track_recent_routing_events(user_id: str = "default", limit: int = 10, **kwargs):
        calls.append(("routing", user_id, limit, kwargs))
        return MemoryStore(tmp_path / "agent.db").recent_routing_events(user_id=user_id, limit=limit, **kwargs)

    def track_recent_retrieval_events(user_id: str = "default", limit: int = 10, **kwargs):
        calls.append(("retrieval", user_id, limit, kwargs))
        return MemoryStore(tmp_path / "agent.db").recent_retrieval_events(user_id=user_id, limit=limit, **kwargs)

    def track_recent_learning_events(user_id: str = "default", limit: int = 10, **kwargs):
        calls.append(("learning", user_id, limit, kwargs))
        return MemoryStore(tmp_path / "agent.db").recent_learning_events(user_id=user_id, limit=limit, **kwargs)

    def track_recent_dedupe_events(user_id: str = "default", limit: int = 10, *, thread_id=None):
        calls.append(("dedupe", user_id, limit, thread_id))
        return MemoryStore(tmp_path / "agent.db").recent_dedupe_events(user_id=user_id, limit=limit, thread_id=thread_id)

    cli_store.long_term_store.profile_store.get_profile = track_get_profile  # type: ignore[method-assign]
    cli_store.long_term_store.transcript_store.thread_messages = track_thread_messages  # type: ignore[method-assign]
    cli_store.long_term_store.audit_store.recent_routing_events = track_recent_routing_events  # type: ignore[method-assign]
    cli_store.long_term_store.audit_store.recent_retrieval_events = track_recent_retrieval_events  # type: ignore[method-assign]
    cli_store.long_term_store.audit_store.recent_learning_events = track_recent_learning_events  # type: ignore[method-assign]
    cli_store.long_term_store.audit_store.recent_dedupe_events = track_recent_dedupe_events  # type: ignore[method-assign]

    cli_store.get_profile()
    cli_store.thread_messages("t1", limit=7)
    cli_store.recent_routing_events(user_id="alice", limit=2, thread_id="t1")
    cli_store.recent_retrieval_events(user_id="alice", limit=3, thread_id="t1")
    cli_store.recent_learning_events(user_id="alice", limit=4, thread_id="t1")
    cli_store.recent_dedupe_events(user_id="alice", limit=5, thread_id="t1")

    assert calls == [
        "profile",
        ("messages", "t1", 7),
        ("routing", "alice", 2, {"thread_id": "t1", "learn": None, "retrieve": None, "reason": None, "text_query": None}),
        ("retrieval", "alice", 3, {"thread_id": "t1"}),
        ("learning", "alice", 4, {"thread_id": "t1"}),
        ("dedupe", "alice", 5, "t1"),
    ]


def test_sqlite_cli_store_delegates_memory_listing_and_delete_to_semantic_adapter(tmp_path):
    memory = MemoryStore(tmp_path / "agent.db")
    cli_store = SqliteCliStore(memory)
    calls = []

    saved = memory.add_memory(
        category="preference",
        content="用户喜欢先给结论再补充原因。",
        importance=4,
        source="conversation",
        user_id="alice",
    )
    memory_id = memory.recent_memories(limit=1, user_id="alice")[0].id
    assert saved is True

    def track_recent_memories(limit: int = 10, user_id: str = "default"):
        calls.append(("recent_memories", limit, user_id))
        return MemoryStore(tmp_path / "agent.db").recent_memories(limit=limit, user_id=user_id)

    def track_delete_memory(memory_id: int, user_id: str = "default"):
        calls.append(("delete_memory", memory_id, user_id))
        return MemoryStore(tmp_path / "agent.db").delete_memory(memory_id, user_id=user_id)

    def track_confirm_memory(memory_id: int, user_id: str = "default"):
        calls.append(("confirm_memory", memory_id, user_id))
        return True

    def track_archive_memory(memory_id: int, user_id: str = "default"):
        calls.append(("archive_memory", memory_id, user_id))
        return True

    def track_dedupe_memories(user_id: str = "default"):
        calls.append(("dedupe_memories", user_id))
        return DedupeResult(removed_count=1, removed_ids=[memory_id], kept_ids=[1])

    cli_store.long_term_store.semantic_memory_store.recent_memories = track_recent_memories  # type: ignore[method-assign]
    cli_store.long_term_store.semantic_memory_store.delete_memory = track_delete_memory  # type: ignore[method-assign]
    cli_store.long_term_store.semantic_memory_store.confirm_memory = track_confirm_memory  # type: ignore[method-assign]
    cli_store.long_term_store.semantic_memory_store.archive_memory = track_archive_memory  # type: ignore[method-assign]
    cli_store.long_term_store.semantic_memory_store.dedupe_memories = track_dedupe_memories  # type: ignore[method-assign]

    recent = cli_store.recent_memories(limit=3, user_id="alice")
    confirmed = cli_store.confirm_memory(memory_id, user_id="alice")
    archived = cli_store.archive_memory(memory_id, user_id="alice")
    deleted = cli_store.delete_memory(memory_id, user_id="alice")
    deduped = cli_store.dedupe_memories(user_id="alice")

    assert recent[0].content == "用户喜欢先给结论再补充原因。"
    assert confirmed is True
    assert archived is True
    assert deleted is True
    assert deduped == DedupeResult(removed_count=1, removed_ids=[memory_id], kept_ids=[1])
    assert calls == [
        ("recent_memories", 3, "alice"),
        ("confirm_memory", memory_id, "alice"),
        ("archive_memory", memory_id, "alice"),
        ("delete_memory", memory_id, "alice"),
        ("dedupe_memories", "alice"),
    ]


def test_sqlite_semantic_memory_store_writes_without_memory_store_add_method(tmp_path):
    memory = MemoryStore(tmp_path / "agent.db")
    store = SqliteSemanticMemoryStore(memory)

    def fail_add_memory(*args, **kwargs):
        raise AssertionError("should not call MemoryStore.add_memory")

    memory.add_memory = fail_add_memory  # type: ignore[method-assign]

    first_saved = store.add_memory(
        category="preference",
        content="用户要求交流时先给出结论，再补充原因。",
        importance=5,
        source="test",
        user_id="alice",
    )
    second_saved = store.add_memory(
        category="preference",
        content="用户偏好回答结构：先给结论，再补充原因。",
        importance=4,
        source="test",
        user_id="alice",
    )

    reloaded = MemoryStore(tmp_path / "agent.db")
    results = reloaded.search_memories("回答偏好", user_id="alice", limit=5)

    assert first_saved is True
    assert second_saved is False
    assert [item.content for item in results] == ["用户要求交流时先给出结论，再补充原因。"]


def test_sqlite_semantic_memory_store_searches_without_memory_store_search_method(tmp_path):
    memory = MemoryStore(tmp_path / "agent.db")
    store = SqliteSemanticMemoryStore(memory)
    memory.add_memory("preference", "用户喜欢回答直接一点，但保留一点温度。", 4, "test", user_id="alice")
    memory.add_memory("fact", "用户正在搭建一个能自我学习的 agent。", 5, "test", user_id="alice")

    def fail_search_memories(*args, **kwargs):
        raise AssertionError("should not call MemoryStore.search_memories")

    memory.search_memories = fail_search_memories  # type: ignore[method-assign]

    results = store.search_memories("这个 agent 应该怎么学习？", limit=2, user_id="alice")

    assert [item.content for item in results] == ["用户正在搭建一个能自我学习的 agent。"]
