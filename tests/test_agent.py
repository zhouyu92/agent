from agent_app.agent import ConversationalAgent, parse_learning_update
from agent_app.config import AgentConfig
from agent_app.memory import MemoryEvolutionResult, MemoryStore


class FakeModel:
    def __init__(self):
        self.calls = []

    def chat(self, messages, temperature=0.7):
        self.calls.append({"messages": messages, "temperature": temperature})
        if temperature == 0.0:
            return """
            {
              "memories": [
                {
                  "category": "preference",
                  "content": "用户喜欢先给结论再解释。",
                  "importance": 4
                }
              ],
              "profile_updates": {
                "style_notes": "回答时先给结论。"
              }
            }
            """
        return "我记住了。"


class DuplicateMemoryModel:
    def chat(self, messages, temperature=0.7):
        if temperature == 0.0:
            return """
            {
              "memories": [
                {
                  "category": "preference",
                  "content": "用户偏好回答结构：先给结论，再补充原因。",
                  "importance": 4
                }
              ],
              "profile_updates": {}
            }
            """
        return "好的。"


class ReflectionModel:
    def __init__(self):
        self.calls = []

    def chat(self, messages, temperature=0.7):
        self.calls.append((messages, temperature))
        return """
        {
          "summary": "用户在多轮对话中稳定偏好先给结论。",
          "memories": [{"category": "preference", "content": "用户喜欢先给结论再解释。", "importance": 4}],
          "profile_updates": {"style_notes": "回答时先给结论。"}
        }
        """


class SummaryModel:
    def __init__(self):
        self.calls = []

    def chat(self, messages, temperature=0.7):
        self.calls.append({"messages": messages, "temperature": temperature})
        if messages[0]["content"].startswith("请将以下对话压缩"):
            return "用户正在推进长期记忆 agent，下一步是测试自动摘要。"
        if temperature == 0.0:
            return '{"memories": [], "profile_updates": {}}'
        return "收到。"


def test_parse_learning_update_accepts_json_inside_markdown_fence():
    raw = """
    ```json
    {
      "memories": [
        {
          "category": "preference",
          "content": "用户喜欢先给结论再解释。",
          "importance": 4
        }
      ],
      "profile_updates": {
        "style_notes": "先短后长。"
      }
    }
    ```
    """

    update = parse_learning_update(raw)

    assert update.memories[0].content == "用户喜欢先给结论再解释。"
    assert update.profile_updates["style_notes"] == "先短后长。"


def test_parse_learning_update_ignores_malformed_json():
    update = parse_learning_update("{not valid json}")

    assert update.memories == []
    assert update.profile_updates == {}


def test_agent_learns_after_reply_and_uses_memory_next_turn(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
    )
    memory = MemoryStore(config.memory_db_path)
    model = FakeModel()
    agent = ConversationalAgent(config, memory, model)

    first_reply = agent.reply("以后回答先给结论。", thread_id="t1")
    second_reply = agent.reply("你还记得我的回答偏好吗？", thread_id="t1")

    assert first_reply == "我记住了。"
    assert second_reply == "我记住了。"
    assert memory.search_memories("回答偏好", limit=1)[0].content == "用户喜欢先给结论再解释。"
    second_chat_messages = model.calls[2]["messages"]
    assert "用户喜欢先给结论再解释。" in second_chat_messages[0]["content"]


def test_agent_injects_thread_summary_into_system_prompt(tmp_path):
    config = AgentConfig(api_key="test-key", base_url="https://example.test/compatible-mode/v1", memory_db_path=tmp_path / "agent.db")
    memory = MemoryStore(config.memory_db_path)
    memory.update_thread_summary("t-summary", "用户正在规划一个长期记忆 agent。", user_id="alice")
    model = FakeModel()

    ConversationalAgent(config, memory, model).reply("请继续。", thread_id="t-summary", user_id="alice")

    assert "用户正在规划一个长期记忆 agent。" in model.calls[0]["messages"][0]["content"]


def test_agent_summarizes_thread_and_persists_result(tmp_path):
    config = AgentConfig(api_key="test-key", base_url="https://example.test/compatible-mode/v1", memory_db_path=tmp_path / "agent.db")
    memory = MemoryStore(config.memory_db_path)
    memory.add_message("t-summary", "user", "我们要做长期记忆。")
    memory.add_message("t-summary", "assistant", "先完成反思闭环。")

    summary = ConversationalAgent(config, memory, FakeModel()).summarize_thread("t-summary", user_id="alice")

    assert summary is not None
    assert memory.get_thread_summary("t-summary", user_id="alice") == summary


def test_agent_automatically_summarizes_new_complete_turns(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        summary_interval=2,
    )
    memory = MemoryStore(config.memory_db_path)
    model = SummaryModel()
    agent = ConversationalAgent(config, memory, model)

    agent.reply("第一轮讨论自动摘要。", thread_id="t-summary", user_id="alice")
    assert memory.get_thread_summary("t-summary", user_id="alice") is None
    agent.reply("第二轮继续讨论自动摘要。", thread_id="t-summary", user_id="alice")

    assert memory.get_thread_summary("t-summary", user_id="alice") == "用户正在推进长期记忆 agent，下一步是测试自动摘要。"
    assert memory.get_thread_summary_last_message_id("t-summary", user_id="alice") == memory.thread_messages("t-summary", limit=10)[-1].id
    summary_calls = [call for call in model.calls if call["messages"][0]["content"].startswith("请将以下对话压缩")]
    assert len(summary_calls) == 1


def test_agent_reflects_unreviewed_learning_events_once(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
    )
    memory = MemoryStore(config.memory_db_path)
    memory.add_learning_event(
        user_id="alice", thread_id="t1", user_text="以后先给结论。", assistant_text="好。", memory_count=1, profile_fields=[]
    )
    memory.add_learning_event(
        user_id="alice", thread_id="t1", user_text="还是先给结论。", assistant_text="明白。", memory_count=1, profile_fields=[]
    )
    model = ReflectionModel()
    agent = ConversationalAgent(config, memory, model)

    result = agent.reflect(thread_id="t1", user_id="alice")
    repeated = agent.reflect(thread_id="t1", user_id="alice")

    assert result.status == "completed"
    assert result.source_event_ids == [1, 2]
    assert result.memory_count == 1
    assert memory.recent_reflection_events(user_id="alice", limit=1)[0].summary == "用户在多轮对话中稳定偏好先给结论。"
    assert repeated.status == "not_ready"
    assert len(model.calls) == 1


def test_agent_automatically_reflects_after_configured_learning_interval(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        reflection_interval=2,
    )
    memory = MemoryStore(config.memory_db_path)
    agent = ConversationalAgent(config, memory, FakeModel())

    agent.reply("以后回答先给结论。", thread_id="t-auto", user_id="alice")
    assert memory.recent_reflection_events(user_id="alice", limit=1) == []
    agent.reply("之后也请先给结论再解释。", thread_id="t-auto", user_id="alice")

    reflection = memory.recent_reflection_events(user_id="alice", limit=1)[0]
    assert reflection.thread_id == "t-auto"
    assert reflection.source_event_ids == [1, 2]


def test_agent_does_not_inject_one_users_memory_into_another_user(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
    )
    memory = MemoryStore(config.memory_db_path)
    model = FakeModel()
    agent = ConversationalAgent(config, memory, model)

    agent.reply("以后回答先给结论。", thread_id="alice-thread", user_id="alice")
    agent.reply("你还记得我的回答偏好吗？", thread_id="bob-thread", user_id="bob")

    bob_chat_messages = model.calls[2]["messages"]
    assert "用户喜欢先给结论再解释。" not in bob_chat_messages[0]["content"]


def test_agent_records_learning_event_after_reply(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
    )
    memory = MemoryStore(config.memory_db_path)
    model = FakeModel()
    agent = ConversationalAgent(config, memory, model)

    agent.reply("以后回答先给结论。", thread_id="t1", user_id="alice")

    events = memory.recent_learning_events(user_id="alice", limit=1)
    assert events[0].thread_id == "t1"
    assert events[0].memory_count == 1
    assert events[0].profile_fields == ["style_notes"]


def test_agent_learning_event_counts_non_ignored_memory_evolution(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
    )
    memory = MemoryStore(config.memory_db_path)
    memory.add_memory(
        "preference",
        "用户要求交流时先给出结论，再补充原因。",
        5,
        "test",
        user_id="alice",
    )
    agent = ConversationalAgent(config, memory, DuplicateMemoryModel())

    agent.reply("继续按我的偏好回答。", thread_id="t1", user_id="alice")

    events = memory.recent_learning_events(user_id="alice", limit=1)
    assert events[0].memory_count == 1


def test_conversational_agent_learn_from_turn_uses_evolve_memory(tmp_path):
    class TrackingMemoryStore(MemoryStore):
        def __init__(self, db_path):
            super().__init__(db_path)
            self.evolve_calls = []

        def evolve_memory(self, *, category, content, importance, source, user_id="default", thread_id=None):
            self.evolve_calls.append((category, content, importance, source, user_id, thread_id))
            return MemoryEvolutionResult(
                action="add",
                candidate_category=category,
                candidate_content=content,
                target_memory_id=None,
                result_memory_id=1,
                reason="new_memory",
            )

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
    )
    memory = TrackingMemoryStore(config.memory_db_path)
    model = FakeModel()
    agent = ConversationalAgent(config, memory, model)

    agent.reply("以后回答先给结论。", thread_id="t1", user_id="alice")

    assert memory.evolve_calls == [
        ("preference", "用户喜欢先给结论再解释。", 4, "conversation", "alice", "t1")
    ]


def test_agent_records_memory_evolution_event_after_reply(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
    )
    memory = MemoryStore(config.memory_db_path)
    model = FakeModel()
    agent = ConversationalAgent(config, memory, model)

    agent.reply("以后回答先给结论。", thread_id="t-evolution", user_id="alice")

    event = memory.recent_memory_evolution_events(user_id="alice", limit=1, thread_id="t-evolution")[0]
    assert event.action == "add"
    assert event.candidate_category == "preference"
    assert event.result_memory_id is not None


def test_classic_agent_persists_routing_audit_for_skipped_turn(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
    )
    memory = MemoryStore(config.memory_db_path)
    agent = ConversationalAgent(config, memory, model=FakeModel())

    agent.reply("谢谢", thread_id="t10", user_id="alice")

    event = memory.recent_routing_events(user_id="alice", limit=1)[0]
    assert event.thread_id == "t10"
    assert event.user_text == "谢谢"
    assert event.should_retrieve is False
    assert event.retrieve_reason == "low_signal"
    assert event.should_learn is False
    assert event.learn_reason == "low_signal"
    assert memory.recent_retrieval_events(user_id="alice", thread_id="t10", limit=1) == []
    assert memory.recent_learning_events(user_id="alice", limit=1) == []


def test_classic_agent_persists_retrieval_audit_for_recalled_turn(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
    )
    memory = MemoryStore(config.memory_db_path)
    memory.add_memory(
        category="preference",
        content="用户喜欢先给结论再补充原因。",
        importance=4,
        source="conversation",
        user_id="alice",
    )
    agent = ConversationalAgent(config, memory, model=FakeModel())

    agent.reply("你还记得我的回答偏好吗？", thread_id="t11", user_id="alice")

    routing_event = memory.recent_routing_events(user_id="alice", limit=1)[0]
    retrieval_event = memory.recent_retrieval_events(user_id="alice", thread_id="t11", limit=1)[0]
    assert routing_event.thread_id == "t11"
    assert routing_event.should_retrieve is True
    assert routing_event.retrieve_reason == "default_retrieve"
    assert routing_event.should_learn is False
    assert routing_event.learn_reason == "recall_turn"
    assert retrieval_event.thread_id == "t11"
    assert retrieval_event.memory_count == 1
    assert len(retrieval_event.memory_ids) == 1
    assert retrieval_event.memory_ids[0] > 0
    assert "先给结论" in retrieval_event.memory_preview
    assert memory.recent_learning_events(user_id="alice", limit=1) == []
