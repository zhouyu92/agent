from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Protocol

from .config import AgentConfig
from .memory import MemoryStore
from .prompts import LEARNING_PROMPT, build_system_prompt


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
    def __init__(self, config: AgentConfig, memory: MemoryStore, model: ChatModel) -> None:
        self.config = config
        self.memory = memory
        self.model = model

    def reply(self, user_text: str, thread_id: str = "default", user_id: str = "default") -> str:
        relevant_memories = self.memory.search_memories(user_text, limit=5, user_id=user_id)
        profile = self.memory.get_profile()
        recent = self.memory.recent_messages(thread_id, self.config.max_recent_turns * 2)
        messages = [{"role": "system", "content": build_system_prompt(profile, relevant_memories)}]
        messages.extend(recent)
        messages.append({"role": "user", "content": user_text})

        assistant_text = self.model.chat(messages)
        self.memory.add_message(thread_id, "user", user_text)
        self.memory.add_message(thread_id, "assistant", assistant_text)
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
                category=str(item.get("category", "general")).strip() or "general",
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
