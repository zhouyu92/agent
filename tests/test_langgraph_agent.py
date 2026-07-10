from agent_app.config import AgentConfig
from agent_app.langgraph_agent import LangGraphAgent
from agent_app.memory import AgentProfile, MemoryEvolutionResult, MemoryItem, MemoryStore
from agent_app.policies import TurnRoutingPolicy
from agent_app.thread_state import CheckpointSnapshot, LangGraphThreadStateStore


class CountingMemoryStore(MemoryStore):
    def __init__(self, db_path):
        super().__init__(db_path)
        self.search_calls = []

    def search_memories(self, query: str, limit: int = 5, user_id: str = "default"):
        self.search_calls.append((query, limit, user_id))
        return super().search_memories(query, limit=limit, user_id=user_id)


class FakeGraphModel:
    def __init__(self):
        self.calls = []
        self.learn_calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        last = messages[-1]
        text = last["content"] if isinstance(last, dict) else last.content
        if "回答偏好" in text:
            return "记得，你喜欢我先给结论再补充原因。"
        return "好的，我之后会先给结论再补充原因。"

    def learn(self, user_text, assistant_text):
        self.learn_calls.append((user_text, assistant_text))
        return {
            "memories": [
                {
                    "category": "preference",
                    "content": "用户喜欢先给结论再补充原因。",
                    "importance": 4,
                }
            ],
            "profile_updates": {"style_notes": "回答时先给结论。"},
        }

    def reflect(self, episodes):
        return {
            "summary": "用户稳定偏好先给结论。",
            "memories": [{"category": "preference", "content": "用户喜欢先给结论再补充原因。", "importance": 4}],
            "profile_updates": {},
        }


class FakeLongTermStore:
    def __init__(self):
        self.profile = AgentProfile(
            identity="像真人一样交流的 agent。",
            style_notes="先给结论。",
            boundaries="不保存敏感信息。",
            updated_at="",
        )
        self.memories = [
            MemoryItem(
                id=7,
                category="preference",
                content="用户喜欢先给结论再补充原因。",
                importance=4,
                source="conversation",
                created_at="",
            )
        ]
        self.search_calls = []
        self.routing_events = []
        self.retrieval_events = []
        self.learning_events = []
        self.reflection_events = []
        self.dedupe_events = []
        self.memory_evolution_events = []
        self.saved_memories = []
        self.profile_updates = []

    def search_memories(self, query: str, limit: int = 5, user_id: str = "default"):
        self.search_calls.append((query, limit, user_id))
        return self.memories[:limit]

    def get_profile(self):
        return self.profile

    def add_memory(self, category: str, content: str, importance: int, source: str, user_id: str = "default"):
        self.saved_memories.append((category, content, importance, source, user_id))
        return True

    def evolve_memory(
        self,
        *,
        category: str,
        content: str,
        importance: int,
        source: str,
        user_id: str = "default",
        thread_id: str | None = None,
    ):
        self.saved_memories.append((category, content, importance, source, user_id, thread_id))
        return MemoryEvolutionResult(
            action="add",
            candidate_category=category,
            candidate_content=content,
            target_memory_id=None,
            result_memory_id=1,
            reason="new_memory",
        )

    def update_profile(self, *, identity=None, style_notes=None, boundaries=None):
        self.profile_updates.append(
            {
                "identity": identity,
                "style_notes": style_notes,
                "boundaries": boundaries,
            }
        )

    def add_learning_event(self, **event):
        self.learning_events.append(event)

    def add_reflection_event(self, **event):
        self.reflection_events.append(event)

    def recent_reflection_events(self, user_id: str = "default", limit: int = 10, *, thread_id=None):
        return self.reflection_events[:limit]

    def add_routing_event(self, **event):
        self.routing_events.append(event)

    def add_retrieval_event(self, **event):
        self.retrieval_events.append(event)

    def add_dedupe_event(self, **event):
        self.dedupe_events.append(event)

    def recent_dedupe_events(self, user_id: str = "default", limit: int = 10, *, thread_id=None):
        return self.dedupe_events[:limit]

    def add_memory_evolution_event(self, **event):
        self.memory_evolution_events.append(event)

    def recent_memory_evolution_events(self, user_id: str = "default", limit: int = 10, *, thread_id=None):
        return self.memory_evolution_events[:limit]


class FakeSemanticMemoryStore:
    def __init__(self):
        self.memories = [
            MemoryItem(
                id=11,
                category="preference",
                content="用户喜欢先给结论再补充原因。",
                importance=4,
                source="conversation",
                created_at="",
            )
        ]
        self.search_calls = []
        self.saved_memories = []

    def search_memories(self, query: str, limit: int = 5, user_id: str = "default"):
        self.search_calls.append((query, limit, user_id))
        return self.memories[:limit]

    def add_memory(self, category: str, content: str, importance: int, source: str, user_id: str = "default"):
        self.saved_memories.append((category, content, importance, source, user_id))
        return True

    def evolve_memory(
        self,
        *,
        category: str,
        content: str,
        importance: int,
        source: str,
        user_id: str = "default",
        thread_id: str | None = None,
    ):
        self.saved_memories.append((category, content, importance, source, user_id, thread_id))
        return MemoryEvolutionResult(
            action="add",
            candidate_category=category,
            candidate_content=content,
            target_memory_id=None,
            result_memory_id=11,
            reason="new_memory",
        )


class FakeProfileStore:
    def __init__(self):
        self.profile = AgentProfile(
            identity="像真人一样交流的 agent。",
            style_notes="先给结论。",
            boundaries="不保存敏感信息。",
            updated_at="",
        )
        self.profile_updates = []

    def get_profile(self):
        return self.profile

    def update_profile(self, *, identity=None, style_notes=None, boundaries=None):
        self.profile_updates.append(
            {
                "identity": identity,
                "style_notes": style_notes,
                "boundaries": boundaries,
            }
        )


class FakeAuditStore:
    def __init__(self):
        self.routing_events = []
        self.retrieval_events = []
        self.learning_events = []
        self.memory_evolution_events = []

    def add_learning_event(self, **event):
        self.learning_events.append(event)

    def add_routing_event(self, **event):
        self.routing_events.append(event)

    def add_retrieval_event(self, **event):
        self.retrieval_events.append(event)

    def add_memory_evolution_event(self, **event):
        self.memory_evolution_events.append(event)


class FakeTranscriptStore:
    def __init__(self):
        self.messages = []

    def add_message(self, thread_id: str, role: str, content: str) -> None:
        self.messages.append((thread_id, role, content))


class FakeThreadStateStore:
    def __init__(self):
        self.messages = [("checkpoint-user", "checkpoint-assistant")]
        self.recorded_turns = []
        self.requested_threads = []
        self.snapshots = []

    def record_turn(self, thread_id: str, user_text: str, assistant_text: str) -> None:
        self.recorded_turns.append((thread_id, user_text, assistant_text))

    def get_thread_messages(self, thread_id: str, user_id: str = "default"):
        self.requested_threads.append((thread_id, user_id))
        return self.messages

    def get_thread_snapshot(self, thread_id: str, user_id: str = "default"):
        self.snapshots.append((thread_id, user_id))
        return CheckpointSnapshot(
            messages=self.messages,
            state_keys=["messages", "retrieved_memories", "routing_decision"],
            message_count=len(self.messages),
            step=3,
            updated_at="2026-07-04T00:00:00+00:00",
            routing_decision={
                "should_retrieve": True,
                "retrieve_reason": "default_retrieve",
                "should_learn": False,
                "learn_reason": "recall_turn",
            },
            retrieved_memories=[
                {
                    "category": "preference",
                    "content": "用户喜欢先给结论再补充原因。",
                    "importance": 4,
                    "source": "conversation",
                }
            ],
        )


class FakeCheckpointStateReader:
    def __init__(self):
        self.requested_threads = []

    def get_thread_messages(self, thread_id: str, user_id: str = "default"):
        self.requested_threads.append(("messages", thread_id, user_id))
        return ["checkpoint-message"]

    def get_thread_snapshot(self, thread_id: str, user_id: str = "default"):
        self.requested_threads.append(("snapshot", thread_id, user_id))
        return CheckpointSnapshot(
            messages=["checkpoint-message"],
            state_keys=["messages"],
            message_count=1,
            step=1,
            updated_at="2026-07-04T00:00:00+00:00",
        )


def test_langgraph_agent_persists_thread_messages_and_long_term_memory(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )
    memory = MemoryStore(config.memory_db_path)
    agent = LangGraphAgent(config, memory, model=FakeGraphModel())

    first = agent.reply("以后回答先给结论，再补充原因。", thread_id="t1", user_id="alice")
    second = agent.reply("你还记得我的回答偏好吗？", thread_id="t1", user_id="alice")

    assert "先给结论" in first
    assert "记得" in second
    assert memory.search_memories("回答偏好", user_id="alice", limit=1)[0].content == "用户喜欢先给结论再补充原因。"
    assert len(agent.get_thread_messages("t1")) == 4
    assert [(message.role, message.content) for message in memory.thread_messages("t1", limit=10)] == [
        ("user", "以后回答先给结论，再补充原因。"),
        ("assistant", "好的，我之后会先给结论再补充原因。"),
        ("user", "你还记得我的回答偏好吗？"),
        ("assistant", "记得，你喜欢我先给结论再补充原因。"),
    ]


def test_langgraph_agent_limits_prompt_history_without_truncating_checkpoint(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
        max_recent_turns=1,
    )
    model = FakeGraphModel()
    agent = LangGraphAgent(config, MemoryStore(config.memory_db_path), model=model)

    agent.reply("第一轮需要学习。", thread_id="t-window", user_id="alice")
    agent.reply("第二轮需要学习。", thread_id="t-window", user_id="alice")
    agent.reply("第三轮需要学习。", thread_id="t-window", user_id="alice")

    prompt_messages = model.calls[-1]
    checkpoint_messages = agent.get_thread_messages("t-window", user_id="alice")
    assert len(prompt_messages) == 3
    assert [message.content for message in prompt_messages[1:]] == ["好的，我之后会先给结论再补充原因。", "第三轮需要学习。"]
    assert len(checkpoint_messages) == 6
    agent.close()


def test_langgraph_agent_summarizes_thread_and_persists_result(tmp_path):
    config = AgentConfig(api_key="test-key", base_url="https://example.test/compatible-mode/v1", memory_db_path=tmp_path / "agent.db", checkpoint_db_path=tmp_path / "checkpoints.db", backend="langgraph")
    memory = MemoryStore(config.memory_db_path)
    memory.add_message("t-summary", "user", "我们要做长期记忆。")
    memory.add_message("t-summary", "assistant", "先完成反思闭环。")
    agent = LangGraphAgent(config, memory, model=FakeGraphModel())

    summary = agent.summarize_thread("t-summary", user_id="alice")

    assert summary is not None
    assert memory.get_thread_summary("t-summary", user_id="alice") == summary
    agent.close()


def test_langgraph_agent_stores_retrieved_memories_in_graph_state(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )
    memory = MemoryStore(config.memory_db_path)
    memory.add_memory(
        category="preference",
        content="用户喜欢先给结论再补充原因。",
        importance=4,
        source="conversation",
        user_id="alice",
    )
    model = FakeGraphModel()
    agent = LangGraphAgent(config, memory, model=model)

    reply = agent.reply("你还记得我的回答偏好吗？", thread_id="t2", user_id="alice")
    state = agent.graph.get_state({"configurable": {"thread_id": "t2", "user_id": "alice"}})

    assert "记得" in reply
    assert state.values["retrieved_memories"][0]["content"] == "用户喜欢先给结论再补充原因。"
    assert "用户喜欢先给结论再补充原因。" in model.calls[-1][0].content


def test_langgraph_agent_keeps_structured_retrieved_memories_in_state(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )
    memory = MemoryStore(config.memory_db_path)
    memory.add_memory(
        category="preference",
        content="用户喜欢先给结论再补充原因。",
        importance=4,
        source="conversation",
        user_id="alice",
    )
    agent = LangGraphAgent(config, memory, model=FakeGraphModel())

    agent.reply("你还记得我的回答偏好吗？", thread_id="t3", user_id="alice")
    state = agent.graph.get_state({"configurable": {"thread_id": "t3", "user_id": "alice"}})
    retrieved = state.values["retrieved_memories"]

    assert retrieved == [
        {
            "category": "preference",
            "content": "用户喜欢先给结论再补充原因。",
            "importance": 4,
            "source": "conversation",
        }
    ]


def test_langgraph_agent_skips_learning_for_sensitive_input(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )
    memory = MemoryStore(config.memory_db_path)
    model = FakeGraphModel()
    agent = LangGraphAgent(config, memory, model=model)

    reply = agent.reply("这是我的 key: sk-abcdefghijklmnopqrstuvwxyz123456", thread_id="t4", user_id="alice")

    assert "先给结论" in reply
    assert model.learn_calls == []
    assert memory.recent_learning_events(user_id="alice") == []


def test_langgraph_agent_skips_learning_for_low_signal_input(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )
    memory = MemoryStore(config.memory_db_path)
    model = FakeGraphModel()
    agent = LangGraphAgent(config, memory, model=model)

    reply = agent.reply("谢谢", thread_id="t5", user_id="alice")

    assert "先给结论" in reply
    assert model.learn_calls == []
    assert memory.recent_learning_events(user_id="alice") == []


def test_langgraph_agent_skips_retrieval_for_low_signal_input(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )
    memory = CountingMemoryStore(config.memory_db_path)
    memory.add_memory(
        category="preference",
        content="用户喜欢先给结论再补充原因。",
        importance=4,
        source="conversation",
        user_id="alice",
    )
    agent = LangGraphAgent(config, memory, model=FakeGraphModel())

    agent.reply("谢谢", thread_id="t6", user_id="alice")
    state = agent.graph.get_state({"configurable": {"thread_id": "t6", "user_id": "alice"}})

    assert state.values["retrieved_memories"] == []
    assert memory.search_calls == []


def test_langgraph_agent_retrieves_but_skips_learning_for_recall_turn(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )
    memory = CountingMemoryStore(config.memory_db_path)
    memory.add_memory(
        category="preference",
        content="用户喜欢先给结论再补充原因。",
        importance=4,
        source="conversation",
        user_id="alice",
    )
    model = FakeGraphModel()
    agent = LangGraphAgent(config, memory, model=model)

    reply = agent.reply("你还记得我的回答偏好吗？", thread_id="t7", user_id="alice")
    state = agent.graph.get_state({"configurable": {"thread_id": "t7", "user_id": "alice"}})

    assert "记得" in reply
    assert state.values["retrieved_memories"][0]["content"] == "用户喜欢先给结论再补充原因。"
    assert memory.search_calls == [("你还记得我的回答偏好吗？", 5, "alice")]
    assert model.learn_calls == []
    assert memory.recent_learning_events(user_id="alice") == []


def test_langgraph_agent_accepts_injected_routing_policy(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )
    memory = CountingMemoryStore(config.memory_db_path)
    model = FakeGraphModel()
    policy = TurnRoutingPolicy()
    agent = LangGraphAgent(config, memory, model=model, routing_policy=policy)

    agent.reply("谢谢", thread_id="t8", user_id="alice")

    assert memory.search_calls == []
    assert model.learn_calls == []


def test_langgraph_agent_exposes_routing_decision_in_state(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )
    memory = CountingMemoryStore(config.memory_db_path)
    model = FakeGraphModel()
    agent = LangGraphAgent(config, memory, model=model)

    agent.reply("谢谢", thread_id="t9", user_id="alice")
    state = agent.graph.get_state({"configurable": {"thread_id": "t9", "user_id": "alice"}})

    assert state.values["routing_decision"] == {
        "should_retrieve": False,
        "retrieve_reason": "low_signal",
        "should_learn": False,
        "learn_reason": "low_signal",
    }


def test_langgraph_agent_persists_routing_audit_for_skipped_turn(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )
    memory = MemoryStore(config.memory_db_path)
    agent = LangGraphAgent(config, memory, model=FakeGraphModel())

    agent.reply("谢谢", thread_id="t10", user_id="alice")
    event = memory.recent_routing_events(user_id="alice", limit=1)[0]

    assert event.thread_id == "t10"
    assert event.user_text == "谢谢"
    assert event.should_retrieve is False
    assert event.retrieve_reason == "low_signal"
    assert event.should_learn is False
    assert event.learn_reason == "low_signal"


def test_langgraph_agent_persists_retrieval_audit_for_recalled_turn(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )
    memory = MemoryStore(config.memory_db_path)
    memory.add_memory(
        category="preference",
        content="用户喜欢先给结论再补充原因。",
        importance=4,
        source="conversation",
        user_id="alice",
    )
    agent = LangGraphAgent(config, memory, model=FakeGraphModel())

    agent.reply("你还记得我的回答偏好吗？", thread_id="t11", user_id="alice")
    event = memory.recent_retrieval_events(user_id="alice", thread_id="t11", limit=1)[0]

    assert event.thread_id == "t11"
    assert event.memory_count == 1
    assert len(event.memory_ids) == 1
    assert event.memory_ids[0] > 0
    assert "先给结论" in event.memory_preview


def test_langgraph_agent_accepts_long_term_store_protocol(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )
    store = FakeLongTermStore()
    agent = LangGraphAgent(config, store, model=FakeGraphModel())

    reply = agent.reply("你还记得我的回答偏好吗？", thread_id="t12", user_id="alice")

    assert "记得" in reply
    assert store.search_calls == [("你还记得我的回答偏好吗？", 5, "alice")]
    assert store.routing_events[0]["thread_id"] == "t12"
    assert store.retrieval_events[0]["memory_ids"] == [7]


def test_langgraph_agent_reflects_learning_events(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )
    memory = MemoryStore(config.memory_db_path)
    memory.add_learning_event(
        user_id="alice", thread_id="t-reflect", user_text="以后先给结论。", assistant_text="好。", memory_count=1, profile_fields=[]
    )
    memory.add_learning_event(
        user_id="alice", thread_id="t-reflect", user_text="还是先给结论。", assistant_text="明白。", memory_count=1, profile_fields=[]
    )
    agent = LangGraphAgent(config, memory, model=FakeGraphModel())

    result = agent.reflect(thread_id="t-reflect", user_id="alice")

    assert result.status == "completed"
    assert result.source_event_ids == [1, 2]
    assert memory.recent_reflection_events(user_id="alice", limit=1)[0].memory_count == 1
    agent.close()


def test_langgraph_agent_automatically_reflects_after_configured_learning_interval(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
        reflection_interval=2,
    )
    memory = MemoryStore(config.memory_db_path)
    agent = LangGraphAgent(config, memory, model=FakeGraphModel())

    agent.reply("以后回答先给结论。", thread_id="t-auto", user_id="alice")
    assert memory.recent_reflection_events(user_id="alice", limit=1) == []
    agent.reply("之后也请先给结论再解释。", thread_id="t-auto", user_id="alice")

    reflection = memory.recent_reflection_events(user_id="alice", limit=1)[0]
    assert reflection.thread_id == "t-auto"
    assert reflection.source_event_ids == [1, 2]
    agent.close()


def test_langgraph_agent_accepts_split_store_dependencies(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )
    semantic_store = FakeSemanticMemoryStore()
    profile_store = FakeProfileStore()
    audit_store = FakeAuditStore()
    transcript_store = FakeTranscriptStore()
    agent = LangGraphAgent(
        config,
        semantic_store,
        model=FakeGraphModel(),
        profile_store=profile_store,
        audit_store=audit_store,
        transcript_store=transcript_store,
    )

    reply = agent.reply("你还记得我的回答偏好吗？", thread_id="t13", user_id="alice")

    assert "记得" in reply
    assert semantic_store.search_calls == [("你还记得我的回答偏好吗？", 5, "alice")]
    assert audit_store.routing_events[0]["thread_id"] == "t13"
    assert audit_store.retrieval_events[0]["memory_ids"] == [11]
    assert transcript_store.messages == [
        ("t13", "user", "你还记得我的回答偏好吗？"),
        ("t13", "assistant", "记得，你喜欢我先给结论再补充原因。"),
    ]


def test_langgraph_agent_delegates_thread_state_to_adapter(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )
    semantic_store = FakeSemanticMemoryStore()
    profile_store = FakeProfileStore()
    audit_store = FakeAuditStore()
    thread_state_store = FakeThreadStateStore()
    agent = LangGraphAgent(
        config,
        semantic_store,
        model=FakeGraphModel(),
        profile_store=profile_store,
        audit_store=audit_store,
        thread_state_store=thread_state_store,
    )

    reply = agent.reply("你还记得我的回答偏好吗？", thread_id="t14", user_id="alice")
    messages = agent.get_thread_messages("t14", user_id="alice")

    assert "记得" in reply
    assert thread_state_store.recorded_turns == [
        ("t14", "你还记得我的回答偏好吗？", "记得，你喜欢我先给结论再补充原因。")
    ]
    assert thread_state_store.requested_threads == [("t14", "alice")]
    assert messages == [("checkpoint-user", "checkpoint-assistant")]


def test_langgraph_thread_state_store_exposes_checkpoint_snapshot():
    reader = FakeCheckpointStateReader()
    store = LangGraphThreadStateStore(reader)

    snapshot = store.get_thread_snapshot("t-state", user_id="alice")

    assert snapshot.state_keys == ["messages"]
    assert snapshot.message_count == 1
    assert reader.requested_threads == [("snapshot", "t-state", "alice")]


def test_langgraph_agent_inspection_uses_thread_state_store_snapshot(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )
    semantic_store = FakeSemanticMemoryStore()
    profile_store = FakeProfileStore()
    audit_store = FakeAuditStore()
    thread_state_store = FakeThreadStateStore()
    agent = LangGraphAgent(
        config,
        semantic_store,
        model=FakeGraphModel(),
        profile_store=profile_store,
        audit_store=audit_store,
        thread_state_store=thread_state_store,
    )

    inspection = agent.inspect_thread("t-snapshot", user_id="alice")

    assert thread_state_store.snapshots == [("t-snapshot", "alice")]
    assert inspection.checkpoint_state_keys == ["messages", "retrieved_memories", "routing_decision"]
    assert inspection.checkpoint_message_count == 1
    assert inspection.checkpoint_step == 3
    assert inspection.checkpoint_updated_at == "2026-07-04T00:00:00+00:00"


def test_langgraph_agent_inspection_exposes_checkpoint_snapshot_metadata(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )
    memory = MemoryStore(config.memory_db_path)
    agent = LangGraphAgent(config, memory, model=FakeGraphModel())

    agent.reply("谢谢", thread_id="t15", user_id="alice")
    inspection = agent.inspect_thread("t15", user_id="alice")

    assert inspection.checkpoint_available is True
    assert inspection.checkpoint_state_keys == ["messages", "retrieved_memories", "routing_decision"]
    assert inspection.checkpoint_message_count == 2
    assert inspection.checkpoint_step == 1
    assert inspection.checkpoint_updated_at is not None
    assert inspection.checkpoint_routing_decision == {
        "should_retrieve": False,
        "retrieve_reason": "low_signal",
        "should_learn": False,
        "learn_reason": "low_signal",
    }
    assert inspection.checkpoint_retrieved_memories == []


def test_langgraph_agent_inspection_exposes_retrieved_memories_from_checkpoint_state(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )
    memory = MemoryStore(config.memory_db_path)
    memory.add_memory(
        category="preference",
        content="用户喜欢先给结论再补充原因。",
        importance=4,
        source="conversation",
        user_id="alice",
    )
    agent = LangGraphAgent(config, memory, model=FakeGraphModel())

    agent.reply("你还记得我的回答偏好吗？", thread_id="t16", user_id="alice")
    inspection = agent.inspect_thread("t16", user_id="alice")

    assert inspection.checkpoint_routing_decision == {
        "should_retrieve": True,
        "retrieve_reason": "default_retrieve",
        "should_learn": False,
        "learn_reason": "recall_turn",
    }
    assert inspection.checkpoint_retrieved_memories == [
        {
            "category": "preference",
            "content": "用户喜欢先给结论再补充原因。",
            "importance": 4,
            "source": "conversation",
        }
    ]


def test_langgraph_learn_node_uses_shared_evolve_memory_api(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )
    semantic_store = FakeSemanticMemoryStore()
    profile_store = FakeProfileStore()
    audit_store = FakeAuditStore()
    transcript_store = FakeTranscriptStore()
    agent = LangGraphAgent(
        config,
        semantic_store,
        model=FakeGraphModel(),
        profile_store=profile_store,
        audit_store=audit_store,
        transcript_store=transcript_store,
    )

    agent.reply("以后回答先给结论。", thread_id="t-evolve", user_id="alice")

    assert semantic_store.saved_memories == [
        ("preference", "用户喜欢先给结论再补充原因。", 4, "conversation", "alice", "t-evolve")
    ]


def test_langgraph_agent_records_memory_evolution_event(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )
    memory = MemoryStore(config.memory_db_path)
    agent = LangGraphAgent(config, memory, model=FakeGraphModel())

    agent.reply("以后回答先给结论。", thread_id="t-evolution", user_id="alice")

    event = memory.recent_memory_evolution_events(user_id="alice", limit=1, thread_id="t-evolution")[0]
    assert event.action == "add"
    assert event.candidate_category == "preference"
    assert event.result_memory_id is not None
