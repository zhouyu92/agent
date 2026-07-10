from __future__ import annotations

from typing import Protocol

from .config import AgentConfig


class ZillizCollectionClient(Protocol):
    def has_collection(self, collection_name: str) -> bool:
        ...

    def create_collection(self, **kwargs) -> object:
        ...

    def upsert(self, **kwargs) -> object:
        ...

    def search(self, **kwargs) -> object:
        ...

    def delete(self, **kwargs) -> object:
        ...


def build_zilliz_client(config: AgentConfig) -> ZillizCollectionClient:
    if not config.zilliz_uri or not config.zilliz_token:
        raise ValueError("ZILLIZ_URI and ZILLIZ_TOKEN are required.")

    from pymilvus import MilvusClient

    return MilvusClient(uri=config.zilliz_uri, token=config.zilliz_token)


def ensure_zilliz_collection(config: AgentConfig, client: ZillizCollectionClient | None = None) -> bool:
    if not config.zilliz_uri or not config.zilliz_token:
        raise ValueError("ZILLIZ_URI and ZILLIZ_TOKEN are required.")

    resolved_client = client or build_zilliz_client(config)
    collection_name = config.zilliz_collection_name
    if resolved_client.has_collection(collection_name):
        return False

    resolved_client.create_collection(
        collection_name=collection_name,
        dimension=config.embedding_dimension,
        metric_type="COSINE",
        primary_field_name="memory_id",
        vector_field_name="vector",
        auto_id=False,
    )
    return True


def upsert_memory_vector(
    config: AgentConfig,
    *,
    client: ZillizCollectionClient | None = None,
    memory_id: int,
    user_id: str,
    category: str,
    status: str,
    content: str,
    vector: list[float],
) -> None:
    if len(vector) != config.embedding_dimension:
        raise ValueError(f"Vector dimension mismatch: expected {config.embedding_dimension}, got {len(vector)}")

    resolved_client = client or build_zilliz_client(config)
    resolved_client.upsert(
        collection_name=config.zilliz_collection_name,
        data=[
            {
                "memory_id": memory_id,
                "vector": vector,
                "user_id": user_id,
                "category": category,
                "status": status,
                "content": content,
            }
        ],
    )


def search_memory_vectors(
    config: AgentConfig,
    *,
    client: ZillizCollectionClient | None = None,
    user_id: str,
    vector: list[float],
    limit: int,
) -> list[int]:
    if len(vector) != config.embedding_dimension:
        raise ValueError(f"Vector dimension mismatch: expected {config.embedding_dimension}, got {len(vector)}")

    resolved_client = client or build_zilliz_client(config)
    results = resolved_client.search(
        collection_name=config.zilliz_collection_name,
        data=[vector],
        limit=limit,
        filter=f'user_id == "{user_id}" and status == "active"',
        output_fields=["memory_id"],
    )
    memory_ids: list[int] = []
    for hit in results[0] if results else []:
        entity = hit.get("entity", hit)
        memory_ids.append(int(entity["memory_id"]))
    return memory_ids


def delete_memory_vector(
    config: AgentConfig,
    *,
    client: ZillizCollectionClient | None = None,
    memory_id: int,
) -> None:
    resolved_client = client or build_zilliz_client(config)
    resolved_client.delete(
        collection_name=config.zilliz_collection_name,
        ids=[memory_id],
    )


def main() -> None:
    config = AgentConfig.from_env()
    created = ensure_zilliz_collection(config)
    if created:
        print(f"created: {config.zilliz_collection_name}")
    else:
        print(f"exists: {config.zilliz_collection_name}")


if __name__ == "__main__":
    main()
