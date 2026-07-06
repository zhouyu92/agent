from __future__ import annotations

from dataclasses import dataclass

from .agent import ConversationalAgent
from .config import AgentConfig
from .langgraph_agent import LangGraphAgent
from .llm import LangChainQwenClient, QwenClient
from .memory import MemoryStore
from .runtime_agent import ClassicConversationRuntime, ConversationRuntime
from .store import SqliteCliStore, SqliteLongTermStore
from .thread_inspection import LangGraphThreadInspectionBuilder
from .thread_state import (
    CheckpointStateReader,
    LangGraphCheckpointStateReader,
    LangGraphThreadStateStore,
    ThreadStateStore,
)


@dataclass(frozen=True)
class AgentRuntime:
    agent: ConversationRuntime
    cli_store: MemoryStore | SqliteCliStore
    memory_store: MemoryStore
    long_term_store: SqliteLongTermStore | None = None
    thread_state_store: ThreadStateStore | None = None
    checkpoint_state_reader: CheckpointStateReader | None = None


def build_runtime(config: AgentConfig) -> AgentRuntime:
    memory_store = MemoryStore(config.memory_db_path)
    if config.backend == "langgraph":
        model = LangChainQwenClient(config)
        long_term_store = SqliteLongTermStore(memory_store)
        cli_store = SqliteCliStore(memory_store, long_term_store)
        checkpoint_state_reader = LangGraphCheckpointStateReader()
        thread_state_store = LangGraphThreadStateStore(
            checkpoint_state_reader,
            transcript_store=long_term_store.transcript_store,
        )
        inspection_builder = LangGraphThreadInspectionBuilder(cli_store, thread_state_store)

        agent = LangGraphAgent(
            config,
            long_term_store.semantic_memory_store,
            model,
            profile_store=long_term_store.profile_store,
            audit_store=long_term_store.audit_store,
            transcript_store=long_term_store.transcript_store,
            thread_state_store=thread_state_store,
            checkpoint_state_reader=checkpoint_state_reader,
            thread_inspector=inspection_builder.build,
        )
        return AgentRuntime(
            agent=agent,
            cli_store=cli_store,
            memory_store=memory_store,
            long_term_store=long_term_store,
            thread_state_store=thread_state_store,
            checkpoint_state_reader=checkpoint_state_reader,
        )

    model = QwenClient(config)
    agent = ClassicConversationRuntime(ConversationalAgent(config, memory_store, model), memory_store)
    return AgentRuntime(agent=agent, cli_store=memory_store, memory_store=memory_store)
