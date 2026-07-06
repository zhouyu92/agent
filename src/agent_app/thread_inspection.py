from __future__ import annotations

from .runtime_agent import ThreadInspection
from .thread_state import ThreadStateStore


class LangGraphThreadInspectionBuilder:
    def __init__(self, cli_store, thread_state_store: ThreadStateStore) -> None:
        self.cli_store = cli_store
        self.thread_state_store = thread_state_store

    def build(self, thread_id: str, user_id: str = "default") -> ThreadInspection:
        snapshot = self.thread_state_store.get_thread_snapshot(thread_id, user_id=user_id)
        return ThreadInspection(
            thread_id=thread_id,
            transcript_messages=self.cli_store.thread_messages(thread_id, limit=50),
            routing_events=self.cli_store.recent_routing_events(user_id=user_id, limit=20, thread_id=thread_id),
            retrieval_events=self.cli_store.recent_retrieval_events(user_id=user_id, limit=20, thread_id=thread_id),
            learning_events=self.cli_store.recent_learning_events(user_id=user_id, limit=20, thread_id=thread_id),
            dedupe_events=self.cli_store.recent_dedupe_events(user_id=user_id, limit=20, thread_id=thread_id),
            memory_evolution_events=self.cli_store.recent_memory_evolution_events(
                user_id=user_id, limit=20, thread_id=thread_id
            ),
            checkpoint_available=True,
            checkpoint_state_keys=snapshot.state_keys,
            checkpoint_message_count=snapshot.message_count,
            checkpoint_step=snapshot.step,
            checkpoint_updated_at=snapshot.updated_at,
            checkpoint_routing_decision=snapshot.routing_decision,
            checkpoint_retrieved_memories=snapshot.retrieved_memories,
            checkpoint_messages=snapshot.messages,
        )
