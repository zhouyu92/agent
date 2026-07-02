from agent_app.memory import MemoryStore


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
