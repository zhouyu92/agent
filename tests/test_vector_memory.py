from agent_app.config import AgentConfig
from agent_app.memory import MemoryItem
from agent_app.vector_memory import VectorMemoryIndexer, VectorMemorySearcher


class FakeEmbeddingClient:
    def __init__(self):
        self.calls = []

    def embed_text(self, text):
        self.calls.append(text)
        return [0.1, 0.2, 0.3]


class FakeVectorWriter:
    def __init__(self):
        self.calls = []
        self.remove_calls = []

    def upsert_memory_vector(self, *, memory_id, user_id, category, status, content, vector):
        self.calls.append(
            {
                "memory_id": memory_id,
                "user_id": user_id,
                "category": category,
                "status": status,
                "content": content,
                "vector": vector,
            }
        )

    def delete_memory_vector(self, *, memory_id):
        self.remove_calls.append(memory_id)


class FakeVectorSearchClient:
    def __init__(self):
        self.calls = []

    def search_memory_vectors(self, *, user_id, vector, limit):
        self.calls.append({"user_id": user_id, "vector": vector, "limit": limit})
        return [42, 7]


def make_config():
    return AgentConfig(
        api_key="test-key",
        base_url="https://example.invalid/v1",
        embedding_dimension=3,
        zilliz_uri="https://example.zilliz.com.cn",
        zilliz_token="test-token",
    )


def make_memory(content="用户喜欢先给结论。"):
    return MemoryItem(
        id=42,
        category="preference",
        content=content,
        importance=4,
        source="conversation",
        created_at="2026-07-06T00:00:00+00:00",
        status="active",
    )


def test_vector_memory_indexer_embeds_and_upserts_memory():
    embedding_client = FakeEmbeddingClient()
    vector_writer = FakeVectorWriter()
    indexer = VectorMemoryIndexer(make_config(), embedding_client=embedding_client, vector_writer=vector_writer)

    indexed = indexer.index_memory(make_memory(), user_id="alice")

    assert indexed is True
    assert embedding_client.calls == ["用户喜欢先给结论。"]
    assert vector_writer.calls == [
        {
            "memory_id": 42,
            "user_id": "alice",
            "category": "preference",
            "status": "active",
            "content": "用户喜欢先给结论。",
            "vector": [0.1, 0.2, 0.3],
        }
    ]


def test_vector_memory_indexer_skips_blank_memory_content():
    embedding_client = FakeEmbeddingClient()
    vector_writer = FakeVectorWriter()
    indexer = VectorMemoryIndexer(make_config(), embedding_client=embedding_client, vector_writer=vector_writer)

    indexed = indexer.index_memory(make_memory(content="   "), user_id="alice")

    assert indexed is False
    assert embedding_client.calls == []
    assert vector_writer.calls == []


def test_vector_memory_indexer_deletes_memory_vector():
    embedding_client = FakeEmbeddingClient()
    vector_writer = FakeVectorWriter()
    indexer = VectorMemoryIndexer(make_config(), embedding_client=embedding_client, vector_writer=vector_writer)

    indexer.remove_memory(memory_id=42)

    assert vector_writer.remove_calls == [42]
    assert embedding_client.calls == []


def test_vector_memory_searcher_embeds_query_and_returns_memory_ids():
    embedding_client = FakeEmbeddingClient()
    vector_search_client = FakeVectorSearchClient()
    searcher = VectorMemorySearcher(
        make_config(),
        embedding_client=embedding_client,
        vector_search_client=vector_search_client,
    )

    memory_ids = searcher.search_memory_ids("回答重点", user_id="alice", limit=2)

    assert memory_ids == [42, 7]
    assert embedding_client.calls == ["回答重点"]
    assert vector_search_client.calls == [{"user_id": "alice", "vector": [0.1, 0.2, 0.3], "limit": 2}]


def test_vector_memory_searcher_skips_blank_query():
    embedding_client = FakeEmbeddingClient()
    vector_search_client = FakeVectorSearchClient()
    searcher = VectorMemorySearcher(
        make_config(),
        embedding_client=embedding_client,
        vector_search_client=vector_search_client,
    )

    assert searcher.search_memory_ids("   ", user_id="alice", limit=2) == []
    assert embedding_client.calls == []
    assert vector_search_client.calls == []
