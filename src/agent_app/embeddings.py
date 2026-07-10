from __future__ import annotations

from typing import Protocol

from openai import OpenAI

from .config import AgentConfig


class EmbeddingsApi(Protocol):
    def create(self, *, model: str, input: str) -> object:
        ...


class OpenAIEmbeddingClient(Protocol):
    embeddings: EmbeddingsApi


class EmbeddingClient:
    def __init__(self, config: AgentConfig, client: OpenAIEmbeddingClient | None = None) -> None:
        self.config = config
        self.client = client or OpenAI(api_key=config.api_key, base_url=config.base_url)

    def embed_text(self, text: str) -> list[float]:
        normalized = text.strip()
        if not normalized:
            return []

        response = self.client.embeddings.create(
            model=self.config.embedding_model,
            input=normalized,
        )
        vector = list(response.data[0].embedding)
        if len(vector) != self.config.embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.config.embedding_dimension}, got {len(vector)}"
            )
        return vector
