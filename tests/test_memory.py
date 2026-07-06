from agent_app.memory import DedupeResult, MemoryStore


def test_memory_store_retrieves_relevant_items(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")
    store.add_memory(
        category="preference",
        content="用户喜欢回答直接一点，但保留一点温度。",
        importance=4,
        source="test",
    )
    store.add_memory(
        category="fact",
        content="用户正在搭建一个能自我学习的 agent。",
        importance=5,
        source="test",
    )

    results = store.search_memories("这个 agent 应该怎么学习？", limit=2)

    assert [item.content for item in results] == ["用户正在搭建一个能自我学习的 agent。"]


def test_agent_profile_can_be_created_and_updated(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")

    profile = store.get_profile()
    assert "像真人一样" in profile.identity

    store.update_profile(style_notes="更坦诚地说明不确定性。")

    updated = store.get_profile()
    assert updated.style_notes == "更坦诚地说明不确定性。"


def test_memory_store_does_not_save_api_keys(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")

    store.add_memory(
        category="fact",
        content="用户的 key 是 sk-1234567890abcdef1234567890abcdef",
        importance=5,
        source="test",
    )

    assert store.search_memories("key", limit=5) == []


def test_recent_memories_returns_latest_items_first(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")
    store.add_memory("fact", "第一条记忆", 3, "test")
    store.add_memory("fact", "第二条记忆", 4, "test")

    results = store.recent_memories(limit=1)

    assert [item.content for item in results] == ["第二条记忆"]


def test_recent_memories_exclude_superseded_by_default(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")
    repository = store.semantic_store.repository
    repository.insert_memory(
        user_id="alice",
        category="preference",
        content="用户喜欢先给结论。",
        importance=4,
        source="conversation",
        created_at="2026-07-06T00:00:00+00:00",
    )
    memory_id = store.recent_memories(user_id="alice", limit=1)[0].id

    repository.mark_memory_superseded(memory_id, user_id="alice")

    assert store.recent_memories(user_id="alice", limit=10) == []


def test_evolve_memory_adds_new_memory_when_no_active_match(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")

    result = store.semantic_store.evolve_memory(
        category="fact",
        content="用户正在搭建一个持续学习 agent。",
        importance=4,
        source="conversation",
        user_id="alice",
    )

    assert result.action == "add"
    assert result.target_memory_id is None
    assert result.result_memory_id is not None


def test_evolve_memory_reinforces_existing_memory(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")
    repository = store.semantic_store.repository
    repository.insert_memory(
        user_id="alice",
        category="preference",
        content="用户喜欢先给结论。",
        importance=3,
        source="conversation",
        created_at="2026-07-06T00:00:00+00:00",
    )
    memory_id = store.recent_memories(user_id="alice", limit=1)[0].id

    result = store.semantic_store.evolve_memory(
        category="preference",
        content="用户还是喜欢先给结论。",
        importance=3,
        source="conversation",
        user_id="alice",
    )
    refreshed = store.recent_memories(user_id="alice", limit=10)[0]

    assert result.action == "reinforce"
    assert result.target_memory_id == memory_id
    assert result.result_memory_id == memory_id
    assert refreshed.id == memory_id


def test_evolve_memory_revises_existing_memory(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")
    repository = store.semantic_store.repository
    repository.insert_memory(
        user_id="alice",
        category="preference",
        content="用户喜欢详细铺垫后再给结论。",
        importance=3,
        source="conversation",
        created_at="2026-07-06T00:00:00+00:00",
    )
    old_id = store.recent_memories(user_id="alice", limit=1)[0].id

    result = store.semantic_store.evolve_memory(
        category="preference",
        content="以后回答不是先铺垫，而是先给结论。",
        importance=4,
        source="conversation",
        user_id="alice",
    )
    all_memories = store.semantic_store.repository.list_memories("alice")

    assert result.action == "revise"
    assert result.target_memory_id == old_id
    assert result.result_memory_id is not None
    assert any(row["id"] == old_id and row["status"] == "superseded" for row in all_memories)


def test_evolve_memory_ignores_low_value_candidate(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")

    result = store.semantic_store.evolve_memory(
        category="general",
        content="好的",
        importance=1,
        source="conversation",
        user_id="alice",
    )

    assert result.action == "ignore"
    assert store.recent_memories(user_id="alice", limit=10) == []


def test_evolve_memory_ignores_identical_candidate(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")
    repository = store.semantic_store.repository
    repository.insert_memory(
        user_id="alice",
        category="preference",
        content="用户喜欢先给结论。",
        importance=3,
        source="conversation",
        created_at="2026-07-06T00:00:00+00:00",
    )
    memory_id = store.recent_memories(user_id="alice", limit=1)[0].id

    result = store.semantic_store.evolve_memory(
        category="preference",
        content="用户喜欢先给结论。",
        importance=3,
        source="conversation",
        user_id="alice",
    )

    assert result.action == "ignore"
    assert result.reason == "no_new_information"
    assert result.target_memory_id == memory_id
    assert result.result_memory_id is None
    assert len(store.recent_memories(user_id="alice", limit=10)) == 1


def test_search_memories_excludes_superseded_items(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")
    repository = store.semantic_store.repository
    repository.insert_memory(
        user_id="alice",
        category="preference",
        content="用户喜欢先铺垫。",
        importance=3,
        source="conversation",
        created_at="2026-07-06T00:00:00+00:00",
    )
    old_id = store.recent_memories(user_id="alice", limit=1)[0].id
    repository.mark_memory_superseded(old_id, user_id="alice")
    repository.insert_revision_memory(
        user_id="alice",
        category="preference",
        content="用户喜欢先给结论。",
        importance=4,
        source="conversation",
        created_at="2026-07-06T00:01:00+00:00",
        supersedes_memory_id=old_id,
    )

    results = store.search_memories("结论", user_id="alice", limit=5)

    assert [item.content for item in results] == ["用户喜欢先给结论。"]


def test_memories_are_isolated_by_user_id(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")
    store.add_memory("preference", "Alice 喜欢先给结论。", 4, "test", user_id="alice")
    store.add_memory("preference", "Bob 喜欢详细推演。", 4, "test", user_id="bob")

    alice_results = store.search_memories("回答偏好", limit=5, user_id="alice")
    bob_results = store.search_memories("回答偏好", limit=5, user_id="bob")

    assert [item.content for item in alice_results] == ["Alice 喜欢先给结论。"]
    assert [item.content for item in bob_results] == ["Bob 喜欢详细推演。"]


def test_learning_events_are_recorded_per_user(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")

    store.add_learning_event(
        user_id="alice",
        thread_id="thread-1",
        user_text="以后回答先给结论。",
        assistant_text="我记住了。",
        memory_count=1,
        profile_fields=["style_notes"],
    )
    store.add_learning_event(
        user_id="bob",
        thread_id="thread-2",
        user_text="我喜欢详细推演。",
        assistant_text="明白。",
        memory_count=1,
        profile_fields=[],
    )

    events = store.recent_learning_events(user_id="alice", limit=5)

    assert len(events) == 1
    assert events[0].thread_id == "thread-1"
    assert events[0].memory_count == 1
    assert events[0].profile_fields == ["style_notes"]


def test_delete_memory_only_deletes_current_users_item(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")
    store.add_memory("preference", "Alice 喜欢先给结论。", 4, "test", user_id="alice")
    store.add_memory("preference", "Bob 喜欢先给结论。", 4, "test", user_id="bob")
    alice_memory = store.search_memories("回答偏好", user_id="alice", limit=1)[0]

    deleted = store.delete_memory(alice_memory.id, user_id="alice")

    assert deleted is True
    assert store.search_memories("回答偏好", user_id="alice", limit=5) == []
    assert [item.content for item in store.search_memories("回答偏好", user_id="bob", limit=5)] == [
        "Bob 喜欢先给结论。"
    ]


def test_messages_redact_api_keys_before_storage(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")

    store.add_message("thread-1", "user", "我的 key 是 sk-1234567890abcdef1234567890abcdef")

    messages = store.recent_messages("thread-1", limit=1)
    assert "sk-1234567890abcdef1234567890abcdef" not in messages[0]["content"]
    assert "[REDACTED_SECRET]" in messages[0]["content"]


def test_learning_events_redact_api_keys_before_storage(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")

    store.add_learning_event(
        user_id="alice",
        thread_id="thread-1",
        user_text="我的 key 是 sk-1234567890abcdef1234567890abcdef",
        assistant_text="我不会保存这个 key。",
        memory_count=0,
        profile_fields=[],
    )

    event = store.recent_learning_events(user_id="alice", limit=1)[0]
    assert "sk-1234567890abcdef1234567890abcdef" not in event.user_text
    assert "[REDACTED_SECRET]" in event.user_text


def test_add_memory_skips_similar_memory_for_same_user_and_category(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")
    first_saved = store.add_memory("preference", "用户要求交流时先给出结论，再补充原因。", 5, "test", user_id="alice")

    second_saved = store.add_memory("preference", "用户偏好回答结构：先给结论，再补充原因。", 4, "test", user_id="alice")

    results = store.search_memories("回答偏好", user_id="alice", limit=5)
    assert first_saved is True
    assert second_saved is False
    assert [item.content for item in results] == ["用户要求交流时先给出结论，再补充原因。"]


def test_similar_memory_is_allowed_for_different_users(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")

    store.add_memory("preference", "用户要求交流时先给出结论，再补充原因。", 5, "test", user_id="alice")
    store.add_memory("preference", "用户要求交流时先给出结论，再补充原因。", 5, "test", user_id="bob")

    assert len(store.search_memories("回答偏好", user_id="alice", limit=5)) == 1
    assert len(store.search_memories("回答偏好", user_id="bob", limit=5)) == 1


def test_dedupe_memories_removes_similar_duplicates_for_same_user_and_category(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")
    repository = store.semantic_store.repository
    repository.insert_memory(
        user_id="alice",
        category="preference",
        content="用户要求交流时先给出结论，再补充原因。",
        importance=5,
        source="test",
        created_at="2026-07-04T00:00:00+00:00",
    )
    repository.insert_memory(
        user_id="alice",
        category="preference",
        content="用户偏好回答结构：先给结论，再补充原因。",
        importance=4,
        source="test",
        created_at="2026-07-04T00:01:00+00:00",
    )

    removed = store.dedupe_memories(user_id="alice")
    results = store.search_memories("回答偏好", user_id="alice", limit=5)

    assert removed.removed_count == 1
    assert removed.removed_ids == [2]
    assert removed.kept_ids == [1]
    assert [item.content for item in results] == ["用户要求交流时先给出结论，再补充原因。"]


def test_dedupe_memories_records_audit_event(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")
    repository = store.semantic_store.repository
    repository.insert_memory(
        user_id="alice",
        category="preference",
        content="用户要求交流时先给出结论，再补充原因。",
        importance=5,
        source="test",
        created_at="2026-07-04T00:00:00+00:00",
    )
    repository.insert_memory(
        user_id="alice",
        category="preference",
        content="用户偏好回答结构：先给结论，再补充原因。",
        importance=4,
        source="test",
        created_at="2026-07-04T00:01:00+00:00",
    )

    removed = store.dedupe_memories(user_id="alice", thread_id="t10")
    events = store.recent_dedupe_events(user_id="alice", limit=1, thread_id="t10")

    assert removed == DedupeResult(removed_count=1, removed_ids=[2], kept_ids=[1])
    assert len(events) == 1
    assert events[0].user_id == "alice"
    assert events[0].thread_id == "t10"
    assert events[0].removed_count == 1
    assert events[0].removed_ids == [2]
    assert events[0].kept_ids == [1]


def test_recent_dedupe_events_support_thread_filter(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")

    store.add_dedupe_event(user_id="alice", thread_id="t10", removed_count=1, removed_ids=[3], kept_ids=[1])
    store.add_dedupe_event(user_id="alice", thread_id="t11", removed_count=2, removed_ids=[4, 5], kept_ids=[2])

    events = store.recent_dedupe_events(user_id="alice", limit=5, thread_id="t11")

    assert len(events) == 1
    assert events[0].thread_id == "t11"
    assert events[0].removed_ids == [4, 5]
    assert events[0].kept_ids == [2]


def test_dedupe_memories_keeps_distinct_or_cross_user_memories(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")
    repository = store.semantic_store.repository
    repository.insert_memory(
        user_id="alice",
        category="preference",
        content="用户要求交流时先给出结论，再补充原因。",
        importance=5,
        source="test",
        created_at="2026-07-04T00:00:00+00:00",
    )
    repository.insert_memory(
        user_id="alice",
        category="fact",
        content="用户正在搭建一个会持续学习的 agent。",
        importance=5,
        source="test",
        created_at="2026-07-04T00:01:00+00:00",
    )
    repository.insert_memory(
        user_id="bob",
        category="preference",
        content="用户偏好回答结构：先给结论，再补充原因。",
        importance=4,
        source="test",
        created_at="2026-07-04T00:02:00+00:00",
    )

    removed = store.dedupe_memories(user_id="alice")

    assert removed == DedupeResult(removed_count=0, removed_ids=[], kept_ids=[])
    assert len(store.search_memories("回答偏好", user_id="alice", limit=5)) == 1
    assert len(store.search_memories("回答偏好", user_id="bob", limit=5)) == 1


def test_routing_events_are_recorded_per_user(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")

    store.add_routing_event(
        user_id="alice",
        thread_id="thread-1",
        user_text="谢谢",
        should_retrieve=False,
        retrieve_reason="low_signal",
        should_learn=False,
        learn_reason="low_signal",
    )

    events = store.recent_routing_events(user_id="alice", limit=5)

    assert len(events) == 1
    assert events[0].thread_id == "thread-1"
    assert events[0].should_retrieve is False
    assert events[0].retrieve_reason == "low_signal"
    assert events[0].should_learn is False
    assert events[0].learn_reason == "low_signal"


def test_recent_routing_events_support_filters(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")

    store.add_routing_event(
        user_id="alice",
        thread_id="t10",
        user_text="谢谢",
        should_retrieve=False,
        retrieve_reason="low_signal",
        should_learn=False,
        learn_reason="low_signal",
    )
    store.add_routing_event(
        user_id="alice",
        thread_id="t11",
        user_text="你还记得我的回答偏好吗？",
        should_retrieve=True,
        retrieve_reason="default_retrieve",
        should_learn=False,
        learn_reason="recall_turn",
    )

    assert [event.thread_id for event in store.recent_routing_events(user_id="alice", thread_id="t11")] == ["t11"]
    assert [event.thread_id for event in store.recent_routing_events(user_id="alice", learn=False)] == ["t11", "t10"]
    assert [event.thread_id for event in store.recent_routing_events(user_id="alice", reason="recall_turn")] == ["t11"]
    assert [event.thread_id for event in store.recent_routing_events(user_id="alice", text_query="谢谢")] == ["t10"]


def test_recent_learning_events_support_thread_filter(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")

    store.add_learning_event(
        user_id="alice",
        thread_id="t10",
        user_text="以后回答先给结论。",
        assistant_text="我记住了。",
        memory_count=1,
        profile_fields=["style_notes"],
    )
    store.add_learning_event(
        user_id="alice",
        thread_id="t11",
        user_text="我喜欢详细推演。",
        assistant_text="明白。",
        memory_count=1,
        profile_fields=[],
    )

    assert [event.thread_id for event in store.recent_learning_events(user_id="alice", thread_id="t11")] == ["t11"]


def test_thread_messages_include_timestamps(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")

    store.add_message("t10", "user", "你好")
    store.add_message("t10", "assistant", "你好，我在。")

    messages = store.thread_messages("t10", limit=10)

    assert [message.role for message in messages] == ["user", "assistant"]
    assert [message.content for message in messages] == ["你好", "你好，我在。"]
    assert messages[0].created_at


def test_retrieval_events_are_recorded_per_user(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")

    store.add_retrieval_event(
        user_id="alice",
        thread_id="t10",
        user_text="你还记得我的回答偏好吗？",
        memory_count=1,
        memory_ids=[7],
        memory_preview="用户喜欢先给结论再补充原因。",
    )

    events = store.recent_retrieval_events(user_id="alice", thread_id="t10")

    assert len(events) == 1
    assert events[0].thread_id == "t10"
    assert events[0].memory_count == 1
    assert events[0].memory_ids == [7]
    assert "先给结论" in events[0].memory_preview


def test_memory_store_migrates_retrieval_event_memory_ids_for_existing_db(tmp_path):
    db_path = tmp_path / "agent.db"
    store = MemoryStore(db_path)
    with store._connect() as conn:
        conn.execute("DROP TABLE retrieval_events")
        conn.execute(
            """
            CREATE TABLE retrieval_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                thread_id TEXT NOT NULL,
                user_text TEXT NOT NULL,
                memory_count INTEGER NOT NULL,
                memory_preview TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

    migrated_store = MemoryStore(db_path)
    migrated_store.add_retrieval_event(
        user_id="alice",
        thread_id="t10",
        user_text="你还记得我的回答偏好吗？",
        memory_count=1,
        memory_ids=[3, 9],
        memory_preview="用户喜欢先给结论再补充原因。",
    )

    events = migrated_store.recent_retrieval_events(user_id="alice", thread_id="t10")

    assert len(events) == 1
    assert events[0].memory_ids == [3, 9]


def test_memory_store_delegates_semantic_methods_to_semantic_store(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")
    calls = []

    class TrackingSemanticStore:
        def add_memory(self, category, content, importance, source, user_id="default"):
            calls.append(("add_memory", category, content, importance, source, user_id))
            return True

        def search_memories(self, query, limit=5, user_id="default"):
            calls.append(("search_memories", query, limit, user_id))
            return []

        def recent_memories(self, limit=10, user_id="default"):
            calls.append(("recent_memories", limit, user_id))
            return []

        def delete_memory(self, memory_id, user_id="default"):
            calls.append(("delete_memory", memory_id, user_id))
            return True

        def dedupe_memories(self, user_id="default"):
            calls.append(("dedupe_memories", user_id))
            return DedupeResult(removed_count=2, removed_ids=[7, 9], kept_ids=[3])

    store.semantic_store = TrackingSemanticStore()  # type: ignore[attr-defined]

    assert store.add_memory("fact", "用户正在搭建 agent。", 4, "test", user_id="alice") is True
    assert store.search_memories("agent", limit=2, user_id="alice") == []
    assert store.recent_memories(limit=3, user_id="alice") == []
    assert store.delete_memory(7, user_id="alice") is True
    assert store.dedupe_memories(user_id="alice") == DedupeResult(removed_count=2, removed_ids=[7, 9], kept_ids=[3])
    assert calls == [
        ("add_memory", "fact", "用户正在搭建 agent。", 4, "test", "alice"),
        ("search_memories", "agent", 2, "alice"),
        ("recent_memories", 3, "alice"),
        ("delete_memory", 7, "alice"),
        ("dedupe_memories", "alice"),
    ]


def test_memory_store_delegates_record_methods_to_record_stores(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")
    calls = []

    class TrackingProfileStore:
        def get_profile(self):
            calls.append("get_profile")
            return MemoryStore(tmp_path / "agent.db").get_profile()

        def update_profile(self, *, identity=None, style_notes=None, boundaries=None):
            calls.append(("update_profile", identity, style_notes, boundaries))

    class TrackingAuditStore:
        def add_learning_event(self, **kwargs):
            calls.append(("add_learning_event", kwargs))

        def recent_learning_events(self, user_id="default", limit=10, *, thread_id=None):
            calls.append(("recent_learning_events", user_id, limit, thread_id))
            return []

        def add_routing_event(self, **kwargs):
            calls.append(("add_routing_event", kwargs))

        def recent_routing_events(
            self,
            user_id="default",
            limit=10,
            *,
            thread_id=None,
            learn=None,
            retrieve=None,
            reason=None,
            text_query=None,
        ):
            calls.append(("recent_routing_events", user_id, limit, thread_id, learn, retrieve, reason, text_query))
            return []

        def add_retrieval_event(self, **kwargs):
            calls.append(("add_retrieval_event", kwargs))

        def recent_retrieval_events(self, user_id="default", limit=10, *, thread_id=None):
            calls.append(("recent_retrieval_events", user_id, limit, thread_id))
            return []

        def add_dedupe_event(self, **kwargs):
            calls.append(("add_dedupe_event", kwargs))

        def recent_dedupe_events(self, user_id="default", limit=10, *, thread_id=None):
            calls.append(("recent_dedupe_events", user_id, limit, thread_id))
            return []

    class TrackingTranscriptStore:
        def add_message(self, thread_id, role, content):
            calls.append(("add_message", thread_id, role, content))

        def recent_messages(self, thread_id, limit):
            calls.append(("recent_messages", thread_id, limit))
            return []

        def thread_messages(self, thread_id, limit=50):
            calls.append(("thread_messages", thread_id, limit))
            return []

    store.profile_store = TrackingProfileStore()  # type: ignore[attr-defined]
    store.audit_store = TrackingAuditStore()  # type: ignore[attr-defined]
    store.transcript_store = TrackingTranscriptStore()  # type: ignore[attr-defined]

    store.get_profile()
    store.update_profile(style_notes="更坦诚。")
    store.add_learning_event(
        user_id="alice",
        thread_id="t1",
        user_text="以后回答先给结论。",
        assistant_text="好。",
        memory_count=1,
        profile_fields=["style_notes"],
    )
    assert store.recent_learning_events(user_id="alice", limit=2, thread_id="t1") == []
    store.add_routing_event(
        user_id="alice",
        thread_id="t1",
        user_text="谢谢",
        should_retrieve=False,
        retrieve_reason="low_signal",
        should_learn=False,
        learn_reason="low_signal",
    )
    assert store.recent_routing_events(user_id="alice", limit=3, thread_id="t1", learn=False) == []
    store.add_retrieval_event(
        user_id="alice",
        thread_id="t1",
        user_text="你还记得吗？",
        memory_count=1,
        memory_ids=[3],
        memory_preview="用户喜欢先给结论。",
    )
    assert store.recent_retrieval_events(user_id="alice", limit=4, thread_id="t1") == []
    assert store.recent_dedupe_events(user_id="alice", limit=5, thread_id="t1") == []
    store.add_message("t1", "user", "你好")
    assert store.recent_messages("t1", limit=5) == []
    assert store.thread_messages("t1", limit=6) == []

    assert calls == [
        "get_profile",
        ("update_profile", None, "更坦诚。", None),
        (
            "add_learning_event",
            {
                "user_id": "alice",
                "thread_id": "t1",
                "user_text": "以后回答先给结论。",
                "assistant_text": "好。",
                "memory_count": 1,
                "profile_fields": ["style_notes"],
            },
        ),
        ("recent_learning_events", "alice", 2, "t1"),
        (
            "add_routing_event",
            {
                "user_id": "alice",
                "thread_id": "t1",
                "user_text": "谢谢",
                "should_retrieve": False,
                "retrieve_reason": "low_signal",
                "should_learn": False,
                "learn_reason": "low_signal",
            },
        ),
        ("recent_routing_events", "alice", 3, "t1", False, None, None, None),
        (
            "add_retrieval_event",
            {
                "user_id": "alice",
                "thread_id": "t1",
                "user_text": "你还记得吗？",
                "memory_count": 1,
                "memory_ids": [3],
                "memory_preview": "用户喜欢先给结论。",
            },
        ),
        ("recent_retrieval_events", "alice", 4, "t1"),
        (
            "recent_dedupe_events",
            "alice",
            5,
            "t1",
        ),
        ("add_message", "t1", "user", "你好"),
        ("recent_messages", "t1", 5),
        ("thread_messages", "t1", 6),
    ]
