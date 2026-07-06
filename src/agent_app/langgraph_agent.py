from __future__ import annotations

from typing import Annotated, Protocol, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .agent import parse_learning_update
from .config import AgentConfig
from .memory import MemoryItem
from .policies import TurnRoutingPolicy
from .prompts import LEARNING_PROMPT, build_system_prompt
from .runtime_agent import ThreadInspection
from .store import AuditStore, MemoryProfileStore, SemanticMemoryStore, ThreadTranscriptStore
from .thread_state import (
    CheckpointStateReader,
    LangGraphCheckpointStateReader,
    LangGraphThreadStateStore,
    ThreadStateStore,
)


class GraphChatModel(Protocol):
    def invoke(self, messages: list[dict[str, str]] | list[BaseMessage]) -> str | AIMessage:
        ...


class RetrievedMemory(TypedDict):
    category: str
    content: str
    importance: int
    source: str


class RoutingDecisionState(TypedDict):
    should_retrieve: bool
    retrieve_reason: str
    should_learn: bool
    learn_reason: str


class LangGraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_memories: list[RetrievedMemory]
    routing_decision: RoutingDecisionState


class LangGraphAgent:
    def __init__(
        self,
        config: AgentConfig,
        semantic_memory_store: SemanticMemoryStore,
        model: GraphChatModel,
        profile_store: MemoryProfileStore | None = None,
        audit_store: AuditStore | None = None,
        transcript_store: ThreadTranscriptStore | None = None,
        thread_state_store: ThreadStateStore | None = None,
        checkpoint_state_reader: CheckpointStateReader | None = None,
        thread_inspector=None,
        routing_policy: TurnRoutingPolicy | None = None,
    ) -> None:
        self.config = config
        self.semantic_memory_store = semantic_memory_store
        self.profile_store = profile_store or self._coerce_profile_store(semantic_memory_store)
        self.audit_store = audit_store or self._coerce_audit_store(semantic_memory_store)
        self.transcript_store = transcript_store or self._coerce_transcript_store(semantic_memory_store)
        self.model = model
        self.routing_policy = routing_policy or TurnRoutingPolicy()
        config.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpointer_cm = SqliteSaver.from_conn_string(str(config.checkpoint_db_path))
        self.checkpointer = self._checkpointer_cm.__enter__()
        self.graph = self._build_graph()
        self.checkpoint_state_reader = checkpoint_state_reader or LangGraphCheckpointStateReader()
        if isinstance(self.checkpoint_state_reader, LangGraphCheckpointStateReader):
            self.checkpoint_state_reader.bind_graph(self.graph)
        self.thread_state_store = thread_state_store or LangGraphThreadStateStore(
            self.checkpoint_state_reader,
            transcript_store=self.transcript_store,
        )
        self.thread_inspector = thread_inspector

    def _build_graph(self):
        builder = StateGraph(LangGraphState)
        builder.add_node("retrieve", self._retrieve_node)
        builder.add_node("respond", self._respond_node)
        builder.add_node("learn", self._learn_node)
        builder.add_conditional_edges(START, self._route_before_retrieve, {"retrieve": "retrieve", "respond": "respond"})
        builder.add_edge("retrieve", "respond")
        builder.add_conditional_edges("respond", self._route_after_respond, {"learn": "learn", "end": END})
        builder.add_edge("learn", END)
        return builder.compile(checkpointer=self.checkpointer)

    def reply(self, user_text: str, thread_id: str = "default", user_id: str = "default") -> str:
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        decision = self.routing_policy.evaluate(user_text).to_state()
        self.audit_store.add_routing_event(
            user_id=user_id,
            thread_id=thread_id,
            user_text=user_text,
            should_retrieve=decision["should_retrieve"],
            retrieve_reason=decision["retrieve_reason"],
            should_learn=decision["should_learn"],
            learn_reason=decision["learn_reason"],
        )
        result = self.graph.invoke(
            {"messages": [HumanMessage(content=user_text)], "retrieved_memories": [], "routing_decision": decision},
            config,
        )
        final_message = result["messages"][-1]
        assistant_text = final_message.content if isinstance(final_message, AIMessage) else str(final_message)
        self.thread_state_store.record_turn(thread_id, user_text, assistant_text)
        return assistant_text

    def get_thread_messages(self, thread_id: str, user_id: str = "default") -> list[BaseMessage]:
        return self.thread_state_store.get_thread_messages(thread_id, user_id=user_id)

    def inspect_thread(self, thread_id: str, user_id: str = "default") -> ThreadInspection:
        if self.thread_inspector is not None:
            return self.thread_inspector(thread_id, user_id)
        snapshot = self.thread_state_store.get_thread_snapshot(thread_id, user_id=user_id)
        return ThreadInspection(
            thread_id=thread_id,
            checkpoint_available=True,
            checkpoint_state_keys=snapshot.state_keys,
            checkpoint_message_count=snapshot.message_count,
            checkpoint_step=snapshot.step,
            checkpoint_updated_at=snapshot.updated_at,
            checkpoint_routing_decision=snapshot.routing_decision,
            checkpoint_retrieved_memories=snapshot.retrieved_memories,
            checkpoint_messages=snapshot.messages,
        )

    def close(self) -> None:
        self._checkpointer_cm.__exit__(None, None, None)

    def _retrieve_node(self, state: LangGraphState, config) -> dict[str, list[RetrievedMemory]]:
        user_id = config["configurable"].get("user_id", "default")
        thread_id = config["configurable"].get("thread_id", "default")
        last_human = self._last_human_message(state["messages"])
        relevant_memories = self.semantic_memory_store.search_memories(last_human.content, limit=5, user_id=user_id)
        self.audit_store.add_retrieval_event(
            user_id=user_id,
            thread_id=thread_id,
            user_text=last_human.content,
            memory_count=len(relevant_memories),
            memory_ids=[item.id for item in relevant_memories],
            memory_preview=" | ".join(item.content for item in relevant_memories[:3]),
        )
        return {
            "retrieved_memories": [
                {
                    "category": item.category,
                    "content": item.content,
                    "importance": item.importance,
                    "source": item.source,
                }
                for item in relevant_memories
            ]
        }

    def _respond_node(self, state: LangGraphState, config) -> dict[str, list[AIMessage]]:
        relevant_memories = [
            MemoryItem(
                id=0,
                category=item["category"],
                content=item["content"],
                importance=item["importance"],
                source=item["source"],
                created_at="",
            )
            for item in state.get("retrieved_memories", [])
        ]
        system_prompt = build_system_prompt(self.profile_store.get_profile(), relevant_memories)
        prompt_messages = [SystemMessage(content=system_prompt), *state["messages"]]
        response = self.model.invoke(prompt_messages)
        ai_message = response if isinstance(response, AIMessage) else AIMessage(content=str(response))
        return {"messages": [ai_message]}

    def _learn_node(self, state: LangGraphState, config) -> dict:
        user_id = config["configurable"].get("user_id", "default")
        thread_id = config["configurable"].get("thread_id", "default")
        human_message = self._last_human_message(state["messages"])
        ai_message = self._last_ai_message(state["messages"])

        update = self._learn_update(human_message.content, ai_message.content)
        saved_memory_count = 0
        for item in update.memories:
            saved = self.semantic_memory_store.evolve_memory(
                category=item.category,
                content=item.content,
                importance=item.importance,
                source="conversation",
                user_id=user_id,
                thread_id=thread_id,
            )
            if self.semantic_memory_store is not self.audit_store:
                self.audit_store.add_memory_evolution_event(
                    user_id=user_id,
                    thread_id=thread_id,
                    action=saved.action,
                    candidate_category=saved.candidate_category,
                    candidate_content=saved.candidate_content,
                    target_memory_id=saved.target_memory_id,
                    result_memory_id=saved.result_memory_id,
                    reason=saved.reason,
                )
            if saved.action != "ignore":
                saved_memory_count += 1

        allowed = {"identity", "style_notes", "boundaries"}
        profile_updates = {key: value for key, value in update.profile_updates.items() if key in allowed}
        if profile_updates:
            self.profile_store.update_profile(**profile_updates)

        self.audit_store.add_learning_event(
            user_id=user_id,
            thread_id=thread_id,
            user_text=human_message.content,
            assistant_text=ai_message.content,
            memory_count=saved_memory_count,
            profile_fields=sorted(profile_updates),
        )
        return {}

    @staticmethod
    def _coerce_profile_store(store: SemanticMemoryStore) -> MemoryProfileStore:
        if isinstance(store, MemoryProfileStore):
            return store
        raise TypeError("profile_store is required when semantic_memory_store does not implement MemoryProfileStore.")

    @staticmethod
    def _coerce_audit_store(store: SemanticMemoryStore) -> AuditStore:
        if isinstance(store, AuditStore):
            return store
        raise TypeError("audit_store is required when semantic_memory_store does not implement AuditStore.")

    @staticmethod
    def _coerce_transcript_store(store: SemanticMemoryStore) -> ThreadTranscriptStore | None:
        if isinstance(store, ThreadTranscriptStore):
            return store
        return None

    def _route_after_respond(self, state: LangGraphState) -> str:
        if not state["routing_decision"]["should_learn"]:
            return "end"
        return "learn"

    def _route_before_retrieve(self, state: LangGraphState) -> str:
        if not state["routing_decision"]["should_retrieve"]:
            return "respond"
        return "retrieve"

    def _learn_update(self, user_text: str, assistant_text: str):
        if hasattr(self.model, "learn"):
            raw = getattr(self.model, "learn")(user_text, assistant_text)
            if isinstance(raw, dict):
                import json

                return parse_learning_update(json.dumps(raw, ensure_ascii=False))
        raw = self.model.invoke(
            [
                {"role": "system", "content": LEARNING_PROMPT},
                {"role": "user", "content": f"用户: {user_text}\n\nagent: {assistant_text}"},
            ]
        )
        content = raw.content if isinstance(raw, AIMessage) else str(raw)
        return parse_learning_update(content)

    @staticmethod
    def _last_human_message(messages: list[BaseMessage]) -> HumanMessage:
        for message in reversed(messages):
            if isinstance(message, HumanMessage):
                return message
        raise ValueError("No human message found in thread state.")

    @staticmethod
    def _last_ai_message(messages: list[BaseMessage]) -> AIMessage:
        for message in reversed(messages):
            if isinstance(message, AIMessage):
                return message
        raise ValueError("No AI message found in thread state.")
