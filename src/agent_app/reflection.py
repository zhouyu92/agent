from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Protocol

from .memory import MemoryEvolutionResult
from .prompts import REFLECTION_PROMPT


class ReflectionAuditStore(Protocol):
    def recent_learning_events(self, user_id: str, limit: int, *, thread_id: str | None = None):
        ...

    def recent_reflection_events(self, user_id: str, limit: int, *, thread_id: str | None = None):
        ...

    def add_reflection_event(self, **kwargs) -> None:
        ...


@dataclass(frozen=True)
class ReflectionRunResult:
    status: str
    source_event_ids: list[int]
    memory_count: int = 0
    profile_fields: list[str] | None = None
    summary: str = ""


def run_reflection(
    *,
    audit_store: ReflectionAuditStore,
    evolve_memory: Callable[[str, str, int], MemoryEvolutionResult],
    update_profile: Callable[..., None],
    model_call: Callable[[str, str], str],
    user_id: str,
    thread_id: str,
    limit: int = 5,
    min_episode_count: int = 2,
) -> ReflectionRunResult:
    prior_reflections = audit_store.recent_reflection_events(user_id=user_id, limit=100, thread_id=thread_id)
    used_event_ids = {event_id for event in prior_reflections for event_id in event.source_event_ids}
    recent_events = audit_store.recent_learning_events(user_id=user_id, limit=100, thread_id=thread_id)
    source_events = [event for event in reversed(recent_events) if event.id not in used_event_ids][:limit]
    if len(source_events) < min_episode_count:
        return ReflectionRunResult(status="not_ready", source_event_ids=[event.id for event in source_events])

    raw = model_call(REFLECTION_PROMPT, _format_episodes(source_events))
    parsed = _parse_reflection(raw)
    if parsed is None:
        return ReflectionRunResult(status="invalid_output", source_event_ids=[event.id for event in source_events])
    update, summary = parsed
    memory_count = sum(
        1
        for item in update.memories
        if evolve_memory(item.category, item.content, item.importance).action != "ignore"
    )
    allowed = {"identity", "style_notes", "boundaries"}
    profile_updates = {key: value for key, value in update.profile_updates.items() if key in allowed}
    if profile_updates:
        update_profile(**profile_updates)
    source_event_ids = [event.id for event in source_events]
    audit_store.add_reflection_event(
        user_id=user_id,
        thread_id=thread_id,
        source_event_ids=source_event_ids,
        summary=summary,
        memory_count=memory_count,
        profile_fields=sorted(profile_updates),
    )
    return ReflectionRunResult(
        status="completed",
        source_event_ids=source_event_ids,
        memory_count=memory_count,
        profile_fields=sorted(profile_updates),
        summary=summary,
    )


def _format_episodes(events) -> str:
    return "\n\n".join(
        f"Episode #{event.id}\n用户: {event.user_text}\nagent: {event.assistant_text}" for event in events
    )


def _parse_reflection(raw: str):
    payload = _extract_json(raw)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    summary = str(data.get("summary", "")).strip() or "No durable cross-turn signal identified."
    from .agent import parse_learning_update

    return parse_learning_update(payload), summary


def _extract_json(raw: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    return raw[start : end + 1] if start != -1 and end >= start else ""
