import pytest

from agent_app.config import AgentConfig
from agent_app.embeddings import EmbeddingClient


class FakeEmbeddingsApi:
    def __init__(self, vector):
        self.vector = vector
        self.calls = []

    def create(self, *, model, input):
        self.calls.append({"model": model, "input": input})
        return type(
            "EmbeddingResponse",
            (),
            {"data": [type("EmbeddingData", (), {"embedding": self.vector})()]},
        )()


class FakeOpenAIClient:
    def __init__(self, vector):
        self.embeddings = FakeEmbeddingsApi(vector)


def make_config(**overrides):
    values = {
        "api_key": "test-key",
        "base_url": "https://example.invalid/v1",
        "embedding_model": "text-embedding-v4",
        "embedding_dimension": 3,
    }
    values.update(overrides)
    return AgentConfig(**values)


def test_embedding_client_embeds_text_with_configured_model():
    openai_client = FakeOpenAIClient([0.1, 0.2, 0.3])
    client = EmbeddingClient(make_config(), client=openai_client)

    vector = client.embed_text("用户喜欢先给结论。")

    assert vector == [0.1, 0.2, 0.3]
    assert openai_client.embeddings.calls == [
        {"model": "text-embedding-v4", "input": "用户喜欢先给结论。"}
    ]


def test_embedding_client_rejects_unexpected_dimension():
    openai_client = FakeOpenAIClient([0.1, 0.2])
    client = EmbeddingClient(make_config(embedding_dimension=3), client=openai_client)

    with pytest.raises(ValueError, match="Embedding dimension mismatch"):
        client.embed_text("用户喜欢先给结论。")


def test_embedding_client_ignores_blank_text():
    openai_client = FakeOpenAIClient([0.1, 0.2, 0.3])
    client = EmbeddingClient(make_config(), client=openai_client)

    assert client.embed_text("   ") == []
    assert openai_client.embeddings.calls == []
