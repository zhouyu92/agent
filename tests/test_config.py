import os

import pytest

from agent_app.config import AgentConfig


def test_config_builds_beijing_base_url_from_workspace_id(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "ws-123")
    monkeypatch.setenv("AGENT_USER_ID", "alice")
    monkeypatch.setenv("AGENT_BACKEND", "langgraph")
    monkeypatch.setenv("AGENT_CHECKPOINT_DB", "data/lg-checkpoints.db")
    monkeypatch.setenv("AGENT_REFLECTION_INTERVAL", "3")
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)

    config = AgentConfig.from_env()

    assert config.api_key == "test-key"
    assert config.model == "qwen3.7-max"
    assert config.base_url == "https://ws-123.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
    assert config.user_id == "alice"
    assert config.backend == "langgraph"
    assert str(config.checkpoint_db_path).endswith("data\\lg-checkpoints.db")
    assert config.reflection_interval == 3


def test_config_rejects_reflection_interval_of_one(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "ws-123")
    monkeypatch.setenv("AGENT_REFLECTION_INTERVAL", "1")
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="AGENT_REFLECTION_INTERVAL"):
        AgentConfig.from_env()


def test_config_reads_embedding_and_zilliz_settings(monkeypatch):
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "ws-123")
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-v4")
    monkeypatch.setenv("EMBEDDING_DIMENSION", "1024")
    monkeypatch.setenv("ZILLIZ_URI", "https://example.zilliz.com.cn")
    monkeypatch.setenv("ZILLIZ_TOKEN", "test-token")
    monkeypatch.delenv("ZILLIZ_COLLECTION_NAME", raising=False)
    monkeypatch.delenv("DASHSCOPE_BASE_URL", raising=False)

    config = AgentConfig.from_env()

    assert config.embedding_model == "text-embedding-v4"
    assert config.embedding_dimension == 1024
    assert config.zilliz_uri == "https://example.zilliz.com.cn"
    assert config.zilliz_token == "test-token"
    assert config.zilliz_collection_name == "zy_test_agent"


def test_config_requires_api_key(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    monkeypatch.setenv("DASHSCOPE_WORKSPACE_ID", "ws-123")

    with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
        AgentConfig.from_env()
