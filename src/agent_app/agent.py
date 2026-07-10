from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from .config import AgentConfig
from .memory import MemoryStore
from .policies import TurnRoutingPolicy
from .prompts import LEARNING_PROMPT, build_system_prompt
from .reflection import ReflectionRunResult, run_reflection
from .semantic_memory import normalize_memory_category


class ChatModel(Protocol):
    def chat(self, messages: list[dict[str, str]], temperature: float = 0.7) -> str:
        ...


@dataclass(frozen=True)
class LearnedMemory:
    category: str
    content: str
    importance: int


@dataclass(frozen=True)
class LearningUpdate:
    memories: list[LearnedMemory] = field(default_factory=list)
    profile_updates: dict[str, str] = field(default_factory=dict)


class ConversationalAgent:
    def __init__(
        self,
        config: AgentConfig,
        memory: MemoryStore,
        model: ChatModel,
        routing_policy: TurnRoutingPolicy | None = None,
    ) -> None:
        self.config = config
        self.memory = memory
        self.model = model
        self.routing_policy = routing_policy or TurnRoutingPolicy()

    def reply(self, user_text: str, thread_id: str = "default", user_id: str = "default") -> str:
        decision = self.routing_policy.evaluate(user_text)
        self.memory.add_routing_event(
            user_id=user_id,
            thread_id=thread_id,
            user_text=user_text,
            should_retrieve=decision.should_retrieve,
            retrieve_reason=decision.retrieve_reason,
            should_learn=decision.should_learn,
            learn_reason=decision.learn_reason,
        )

        relevant_memories = []
        if decision.should_retrieve:
            relevant_memories = self.memory.search_memories(user_text, limit=5, user_id=user_id)
            self.memory.add_retrieval_event(
                user_id=user_id,
                thread_id=thread_id,
                user_text=user_text,
                memory_count=len(relevant_memories),
                memory_ids=[item.id for item in relevant_memories],
                memory_preview=" | ".join(item.content for item in relevant_memories[:3]),
            )
        profile = self.memory.get_profile()
        recent = self.memory.recent_messages(thread_id, self.config.max_recent_turns * 2)
        thread_summary = self.memory.get_thread_summary(thread_id, user_id=user_id)
        messages = [{"role": "system", "content": build_system_prompt(profile, relevant_memories, thread_summary)}]
        messages.extend(recent)
        messages.append({"role": "user", "content": user_text})

        assistant_text = self.model.chat(messages)
        self.memory.add_message(thread_id, "user", user_text)
        self.memory.add_message(thread_id, "assistant", assistant_text)
        if decision.should_learn:
            self._learn_from_turn(user_text, assistant_text, thread_id=thread_id, user_id=user_id)
        return assistant_text

    def _learn_from_turn(self, user_text: str, assistant_text: str, thread_id: str, user_id: str) -> None:
        learning_messages = [
            {"role": "system", "content": LEARNING_PROMPT},
            {
                "role": "user",
                "content": f"用户: {user_text}\n\nagent: {assistant_text}",
            },
        ]
        raw = self.model.chat(learning_messages, temperature=0.0)
        update = parse_learning_update(raw)
        saved_memory_count = 0
        for item in update.memories:
            saved = self.memory.evolve_memory(
                category=item.category,
                content=item.content,
                importance=item.importance,
                source="conversation",
                user_id=user_id,
                thread_id=thread_id,
            )
            if saved.action != "ignore":
                saved_memory_count += 1
        allowed = {"identity", "style_notes", "boundaries"}
        profile_updates = {key: value for key, value in update.profile_updates.items() if key in allowed}
        if profile_updates:
            self.memory.update_profile(**profile_updates)
        self.memory.add_learning_event(
            user_id=user_id,
            thread_id=thread_id,
            user_text=user_text,
            assistant_text=assistant_text,
            memory_count=saved_memory_count,
            profile_fields=sorted(profile_updates),
        )

        if self.config.reflection_interval >= 2:
            self.reflect(thread_id=thread_id, user_id=user_id, min_episode_count=self.config.reflection_interval)

    def reflect(
        self,
        thread_id: str = "default",
        user_id: str = "default",
        min_episode_count: int = 2,
    ) -> ReflectionRunResult:
        return run_reflection(
            audit_store=self.memory,
            evolve_memory=lambda category, content, importance: self.memory.evolve_memory(
                category=category,
                content=content,
                importance=importance,
                source="reflection",
                user_id=user_id,
                thread_id=thread_id,
            ),
            update_profile=self.memory.update_profile,
            model_call=lambda system, episodes: self.model.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": episodes}], temperature=0.0
            ),
            user_id=user_id,
            thread_id=thread_id,
            min_episode_count=min_episode_count,
        )


def parse_learning_update(raw: str) -> LearningUpdate:
    try:
        data = json.loads(_extract_json(raw))
    except json.JSONDecodeError:
        return LearningUpdate()
    memories = []
    for item in data.get("memories", []):
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        memories.append(
            LearnedMemory(
                category=normalize_memory_category(str(item.get("category", "general"))),
                content=content,
                importance=int(item.get("importance", 3)),
            )
        )
    profile_updates = {
        str(key): str(value).strip()
        for key, value in data.get("profile_updates", {}).items()
        if str(value).strip()
    }
    return LearningUpdate(memories=memories, profile_updates=profile_updates)


def _extract_json(raw: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return '{"memories": [], "profile_updates": {}}'
    return raw[start : end + 1]
