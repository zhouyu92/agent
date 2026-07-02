from __future__ import annotations

from typing import Annotated, Protocol, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .agent import parse_learning_update
from .config import AgentConfig
from .memory import MemoryStore
from .prompts import LEARNING_PROMPT, build_system_prompt


class GraphChatModel(Protocol):
    def invoke(self, messages: list[dict[str, str]] | list[BaseMessage]) -> str | AIMessage:
        ...


class LangGraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class LangGraphAgent:
    def __init__(self, config: AgentConfig, memory: MemoryStore, model: GraphChatModel) -> None:
        self.config = config
        self.memory = memory
        self.model = model
        config.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._checkpointer_cm = SqliteSaver.from_conn_string(str(config.checkpoint_db_path))
        self.checkpointer = self._checkpointer_cm.__enter__()
        self.graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(LangGraphState)
        builder.add_node("respond", self._respond_node)
        builder.add_node("learn", self._learn_node)
        builder.add_edge(START, "respond")
        builder.add_edge("respond", "learn")
        builder.add_edge("learn", END)
        return builder.compile(checkpointer=self.checkpointer)

    def reply(self, user_text: str, thread_id: str = "default", user_id: str = "default") -> str:
        config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
        result = self.graph.invoke({"messages": [HumanMessage(content=user_text)]}, config)
        final_message = result["messages"][-1]
        return final_message.content if isinstance(final_message, AIMessage) else str(final_message)

    def get_thread_messages(self, thread_id: str, user_id: str = "default") -> list[BaseMessage]:
        snapshot = self.graph.get_state({"configurable": {"thread_id": thread_id, "user_id": user_id}})
        return list(snapshot.values.get("messages", []))

    def close(self) -> None:
        self._checkpointer_cm.__exit__(None, None, None)

    def _respond_node(self, state: LangGraphState, config) -> dict[str, list[AIMessage]]:
        user_id = config["configurable"].get("user_id", "default")
        last_human = self._last_human_message(state["messages"])
        relevant_memories = self.memory.search_memories(last_human.content, limit=5, user_id=user_id)
        system_prompt = build_system_prompt(self.memory.get_profile(), relevant_memories)
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
            saved = self.memory.add_memory(
                category=item.category,
                content=item.content,
                importance=item.importance,
                source="conversation",
                user_id=user_id,
            )
            if saved:
                saved_memory_count += 1

        allowed = {"identity", "style_notes", "boundaries"}
        profile_updates = {key: value for key, value in update.profile_updates.items() if key in allowed}
        if profile_updates:
            self.memory.update_profile(**profile_updates)

        self.memory.add_learning_event(
            user_id=user_id,
            thread_id=thread_id,
            user_text=human_message.content,
            assistant_text=ai_message.content,
            memory_count=saved_memory_count,
            profile_fields=sorted(profile_updates),
        )
        return {}

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
