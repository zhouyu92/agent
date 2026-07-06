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
