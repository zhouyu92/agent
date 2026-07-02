from agent_app.config import AgentConfig
from agent_app.langgraph_agent import LangGraphAgent
from agent_app.memory import MemoryStore


class FakeGraphModel:
    def __init__(self):
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        last = messages[-1]
        text = last["content"] if isinstance(last, dict) else last.content
        if "回答偏好" in text:
            return "记得，你喜欢我先给结论再补充原因。"
        return "好的，我之后会先给结论再补充原因。"

    def learn(self, user_text, assistant_text):
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
