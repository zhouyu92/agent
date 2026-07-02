import os

import pytest

from agent_app.config import AgentConfig


def test_config_builds_beijing_base_url_from_workspace_id(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "ws-123")
    monkeypatch.setenv("AGENT_USER_ID", "alice")
    monkeypatch.setenv("AGENT_BACKEND", "langgraph")
    monkeypatch.setenv("AGENT_CHECKPOINT_DB", "data/lg-checkpoints.db")
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)

    config = AgentConfig.from_env()

    assert config.api_key == "test-key"
    assert config.model == "qwen3.7-max"
    assert config.base_url == "https://ws-123.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    assert config.user_id == "alice"
    assert config.backend == "langgraph"
    assert str(config.checkpoint_db_path).endswith("data\\lg-checkpoints.db")


def test_config_requires_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "ws-123")

    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        AgentConfig.from_env()
