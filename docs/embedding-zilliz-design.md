# Embedding and Zilliz Design

## Goal

Embedding is only used for two long-term memory operations:

- retrieving relevant active memories before a reply
- finding similar existing memories before `add / reinforce / revise / ignore`

Embedding does not replace the memory evolution rules. The rule layer still decides whether a candidate memory should be added, reinforced, revised, or ignored.

## Configuration

Runtime configuration is loaded from environment variables:

- `EMBEDDING_MODEL`: default `text-embedding-v4`
- `EMBEDDING_DIMENSION`: default `1024`
- `ZILLIZ_URI`: Zilliz Cloud endpoint
- `ZILLIZ_TOKEN`: Zilliz Cloud token
- `ZILLIZ_COLLECTION_NAME`: default `zy_test_agent`

Secrets must not be committed. `ZILLIZ_TOKEN` should only live in local environment configuration.

## Storage Boundary

SQLite remains the source of truth for durable memory records:

- memory id
- user id
- category
- content
- importance
- status
- supersession history
- audit events

Zilliz stores vector search material derived from active memory content:

- `memory_id`
- `user_id`
- `category`
- `status`
- embedding vector
- `content`

The application should use Zilliz search results to find candidate memory ids, then load authoritative memory records from SQLite.

## Retrieval Flow

1. Generate an embedding for the user query.
2. Search the Zilliz collection for candidate active memories.
3. Filter by `user_id` and active status.
4. Load matching records from SQLite.
5. Fall back to the current keyword matcher if embedding or Zilliz is unavailable.

The semantic store treats Zilliz results as candidate ids only. It preserves Zilliz ranking order when loading active memories from SQLite. If vector search raises an error or returns no usable active ids, the existing keyword matcher is used.

## Similarity Flow

1. Generate an embedding for the candidate memory content.
2. Search nearby active memories in the same `user_id` and category.
3. Pass the best candidate into the existing evolution decision helper.
4. Keep the existing rule layer responsible for `add / reinforce / revise / ignore`.

Vector search is only candidate recall for similarity matching. Candidates from another category are ignored, and if vector search fails the semantic store falls back to the existing keyword/rule-based matcher.

## Indexing Boundary

`VectorMemoryIndexer` is the small integration point between embeddings and Zilliz:

1. Accept a SQLite-backed memory record.
2. Generate an embedding from the memory content.
3. Upsert `memory_id`, vector, and metadata into Zilliz.

It does not decide whether a memory should exist. It only indexes memories that the long-term memory layer has already accepted.

The semantic store may receive an optional indexer. When configured, successful `add` and `revise` evolution results index the new active memory. Indexing failures must not roll back or block the SQLite memory write.
When a `revise` supersedes an old memory, the old vector is removed from Zilliz by `memory_id` on a best-effort basis. Removal failures also do not roll back the SQLite revision.

Runtime bootstrap creates the indexer only when both `ZILLIZ_URI` and `ZILLIZ_TOKEN` are configured. Without those settings, the agent continues to use the SQLite and keyword-matching path only.
The same condition enables vector indexing and search for both classic and LangGraph long-term memory paths.

`agent-vector-smoke` verifies the live loop by creating a temporary SQLite memory, indexing it into Zilliz, searching it back by vector, and cleaning up both stores.

## Collection

The planned collection name is:

```text
zy_test_agent
```

The initial dimension is `1024`, matching `text-embedding-v4`.

The collection uses quick setup with:

- primary field: `memory_id`
- vector field: `vector`
- metric: `COSINE`
- dynamic metadata fields: `user_id`, `category`, `status`, `content`

## Rollout

1. Add configuration and documentation.
2. Add an optional Zilliz collection bootstrap helper.
3. Add an embedding client abstraction.
4. Add a vector memory indexer for explicit upsert.
5. Add vector search for long-term memory retrieval.
6. Add vector candidate recall for memory evolution similarity matching.
7. Keep keyword matching as fallback until vector behavior is verified.

The embedding client uses the existing OpenAI-compatible `base_url` and API key, but calls the configured embedding model instead of the chat model.
