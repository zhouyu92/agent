from __future__ import annotations

from typing import Protocol

from .config import AgentConfig
from .embeddings import EmbeddingClient
from .memory import MemoryItem
from .zilliz import delete_memory_vector, search_memory_vectors, upsert_memory_vector


class MemoryEmbeddingClient(Protocol):
    def embed_text(self, text: str) -> list[float]:
        ...


class MemoryVectorWriter(Protocol):
    def upsert_memory_vector(
        self,
        *,
        memory_id: int,
        user_id: str,
        category: str,
        status: str,
        content: str,
        vector: list[float],
    ) -> None:
        ...

    def delete_memory_vector(self, *, memory_id: int) -> None:
        ...


class MemoryVectorSearchClient(Protocol):
    def search_memory_vectors(self, *, user_id: str, vector: list[float], limit: int) -> list[int]:
        ...


class ZillizMemoryVectorWriter:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def upsert_memory_vector(
        self,
        *,
        memory_id: int,
        user_id: str,
        category: str,
        status: str,
        content: str,
        vector: list[float],
    ) -> None:
        upsert_memory_vector(
            self.config,
            memory_id=memory_id,
            user_id=user_id,
            category=category,
            status=status,
            content=content,
            vector=vector,
        )

    def delete_memory_vector(self, *, memory_id: int) -> None:
        delete_memory_vector(self.config, memory_id=memory_id)


class ZillizMemoryVectorSearchClient:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def search_memory_vectors(self, *, user_id: str, vector: list[float], limit: int) -> list[int]:
        return search_memory_vectors(
            self.config,
            user_id=user_id,
            vector=vector,
            limit=limit,
        )


class VectorMemoryIndexer:
    def __init__(
        self,
        config: AgentConfig,
        *,
        embedding_client: MemoryEmbeddingClient | None = None,
        vector_writer: MemoryVectorWriter | None = None,
    ) -> None:
        self.embedding_client = embedding_client or EmbeddingClient(config)
        self.vector_writer = vector_writer or ZillizMemoryVectorWriter(config)

    def index_memory(self, memory: MemoryItem, *, user_id: str) -> bool:
        content = memory.content.strip()
        if not content:
            return False

        vector = self.embedding_client.embed_text(content)
        if not vector:
            return False

        self.vector_writer.upsert_memory_vector(
            memory_id=memory.id,
            user_id=user_id,
            category=memory.category,
            status=memory.status,
            content=content,
            vector=vector,
        )
        return True

    def remove_memory(self, *, memory_id: int) -> None:
        self.vector_writer.delete_memory_vector(memory_id=memory_id)


class VectorMemorySearcher:
    def __init__(
        self,
        config: AgentConfig,
        *,
        embedding_client: MemoryEmbeddingClient | None = None,
        vector_search_client: MemoryVectorSearchClient | None = None,
    ) -> None:
        self.embedding_client = embedding_client or EmbeddingClient(config)
        self.vector_search_client = vector_search_client or ZillizMemoryVectorSearchClient(config)

    def search_memory_ids(self, query: str, *, user_id: str, limit: int) -> list[int]:
        normalized = query.strip()
        if not normalized:
            return []

        vector = self.embedding_client.embed_text(normalized)
        if not vector:
            return []

        return self.vector_search_client.search_memory_vectors(
            user_id=user_id,
            vector=vector,
            limit=limit,
        )
