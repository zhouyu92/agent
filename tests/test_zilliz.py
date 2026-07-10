import pytest

from agent_app.config import AgentConfig
from agent_app.zilliz import ensure_zilliz_collection


class FakeMilvusClient:
    def __init__(self, exists=False):
        self.exists = exists
        self.create_calls = []

    def has_collection(self, collection_name):
        return self.exists and collection_name == "zy_test_agent"

    def create_collection(self, **kwargs):
        self.create_calls.append(kwargs)

    def upsert(self, **kwargs):
        self.upsert_call = kwargs

    def search(self, **kwargs):
        self.search_call = kwargs
        return [[{"entity": {"memory_id": 42}}, {"entity": {"memory_id": 7}}]]

    def delete(self, **kwargs):
        self.delete_call = kwargs
        return {"delete_count": 1}


def make_config(**overrides):
    values = {
        "api_key": "test-key",
        "base_url": "https://example.invalid/v1",
        "zilliz_uri": "https://example.zilliz.com.cn",
        "zilliz_token": "test-token",
        "zilliz_collection_name": "zy_test_agent",
        "embedding_dimension": 1024,
    }
    values.update(overrides)
    return AgentConfig(**values)


def test_ensure_zilliz_collection_creates_missing_collection():
    client = FakeMilvusClient(exists=False)

    created = ensure_zilliz_collection(make_config(), client=client)

    assert created is True
    assert client.create_calls == [
        {
            "collection_name": "zy_test_agent",
            "dimension": 1024,
            "metric_type": "COSINE",
            "primary_field_name": "memory_id",
            "vector_field_name": "vector",
            "auto_id": False,
        }
    ]


def test_ensure_zilliz_collection_skips_existing_collection():
    client = FakeMilvusClient(exists=True)

    created = ensure_zilliz_collection(make_config(), client=client)

    assert created is False
    assert client.create_calls == []


def test_ensure_zilliz_collection_requires_zilliz_config():
    client = FakeMilvusClient(exists=False)

    with pytest.raises(ValueError, match="ZILLIZ_URI and ZILLIZ_TOKEN"):
        ensure_zilliz_collection(make_config(zilliz_token=None), client=client)


def test_upsert_memory_vector_writes_memory_metadata():
    from agent_app.zilliz import upsert_memory_vector

    client = FakeMilvusClient(exists=True)

    upsert_memory_vector(
        make_config(embedding_dimension=3),
        client=client,
        memory_id=42,
        user_id="alice",
        category="preference",
        status="active",
        content="用户喜欢先给结论。",
        vector=[0.1, 0.2, 0.3],
    )

    assert client.upsert_call == {
        "collection_name": "zy_test_agent",
        "data": [
            {
                "memory_id": 42,
                "vector": [0.1, 0.2, 0.3],
                "user_id": "alice",
                "category": "preference",
                "status": "active",
                "content": "用户喜欢先给结论。",
            }
        ],
    }


def test_upsert_memory_vector_rejects_wrong_dimension():
    from agent_app.zilliz import upsert_memory_vector

    client = FakeMilvusClient(exists=True)

    with pytest.raises(ValueError, match="Vector dimension mismatch"):
        upsert_memory_vector(
            make_config(embedding_dimension=3),
            client=client,
            memory_id=42,
            user_id="alice",
            category="preference",
            status="active",
            content="用户喜欢先给结论。",
            vector=[0.1, 0.2],
        )


def test_search_memory_vectors_returns_memory_ids():
    from agent_app.zilliz import search_memory_vectors

    client = FakeMilvusClient(exists=True)

    memory_ids = search_memory_vectors(
        make_config(embedding_dimension=3),
        client=client,
        user_id="alice",
        vector=[0.1, 0.2, 0.3],
        limit=2,
    )

    assert memory_ids == [42, 7]
    assert client.search_call == {
        "collection_name": "zy_test_agent",
        "data": [[0.1, 0.2, 0.3]],
        "limit": 2,
        "filter": 'user_id == "alice" and status == "active"',
        "output_fields": ["memory_id"],
    }


def test_delete_memory_vector_deletes_by_memory_id():
    from agent_app.zilliz import delete_memory_vector

    client = FakeMilvusClient(exists=True)

    delete_memory_vector(make_config(), client=client, memory_id=42)

    assert client.delete_call == {
        "collection_name": "zy_test_agent",
        "ids": [42],
    }
