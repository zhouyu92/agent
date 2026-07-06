from __future__ import annotations

from .agent import ConversationalAgent
from .bootstrap import AgentRuntime, build_runtime
from .config import AgentConfig
from .inspection_report import (
    format_audit_timeline,
    format_checkpoint_diff,
    format_checkpoint_messages,
    format_retrieval_comparison,
    format_thread_inspection,
)
from .langgraph_agent import LangGraphAgent
from .memory import (
    AgentProfile,
    DedupeEvent,
    LearningEvent,
    MemoryEvolutionEvent,
    MemoryItem,
    MemoryStore,
    RetrievalEvent,
    RoutingEvent,
    ThreadMessage,
)
from .runtime_agent import ConversationRuntime, ThreadInspection, ThreadInspectionRuntime
from .store import SqliteCliStore


def build_agent(config: AgentConfig) -> tuple[ConversationRuntime, MemoryStore | SqliteCliStore]:
    runtime = build_runtime(config)
    return runtime.agent, runtime.cli_store


def parse_memory_query(query: str) -> tuple[str, dict[str, object], str | None]:
    usage = "Usage: /memories [category=<name>] [importance=<1-5>] [status=<active|superseded>] [query]"
    filters: dict[str, object] = {
        "category": None,
        "importance": None,
        "status": None,
    }
    text_terms: list[str] = []
    for token in query.split():
        lowered = token.lower()
        if lowered.startswith("category="):
            value = token.split("=", 1)[1].strip().lower()
            if not value:
                return "", {}, usage
            filters["category"] = value
            continue
        if lowered.startswith("importance="):
            value = token.split("=", 1)[1]
            if not value.isdigit():
                return "", {}, usage
            filters["importance"] = int(value)
            if not 1 <= filters["importance"] <= 5:
                return "", {}, usage
            continue
        if lowered.startswith("status="):
            value = token.split("=", 1)[1].strip().lower()
            if value not in {"active", "superseded"}:
                return "", {}, usage
            filters["status"] = value
            continue
        if "=" in token:
            return "", {}, usage
        text_terms.append(token)
    active_filters = {key: value for key, value in filters.items() if value is not None}
    return " ".join(text_terms), active_filters, None


def parse_dedupe_query(query: str) -> tuple[dict[str, object], str | None]:
    usage = "Usage: /dedupe-memories [thread=<thread_id>]"
    filters: dict[str, object] = {
        "thread_id": None,
    }
    for token in query.split():
        lowered = token.lower()
        if lowered.startswith("thread="):
            value = token.split("=", 1)[1].strip()
            if not value:
                return {}, usage
            filters["thread_id"] = value
            continue
        if "=" in token:
            return {}, usage
        return {}, usage
    return {key: value for key, value in filters.items() if value is not None}, None


def parse_dedupe_log_query(query: str) -> tuple[dict[str, object], str | None]:
    usage = "Usage: /dedupe-log [thread=<thread_id>] [limit=<n>]"
    filters: dict[str, object] = {
        "thread_id": None,
        "limit": None,
    }
    for token in query.split():
        lowered = token.lower()
        if lowered.startswith("thread="):
            value = token.split("=", 1)[1].strip()
            if not value:
                return {}, usage
            filters["thread_id"] = value
            continue
        if lowered.startswith("limit="):
            value = token.split("=", 1)[1].strip()
            if not value.isdigit():
                return {}, usage
            limit = int(value)
            if limit <= 0:
                return {}, usage
            filters["limit"] = limit
            continue
        if "=" in token:
            return {}, usage
        return {}, usage
    return {key: value for key, value in filters.items() if value is not None}, None


def parse_memory_evolution_log_query(query: str) -> tuple[dict[str, object], str | None]:
    usage = "Usage: /memory-log [thread=<thread_id>] [action=<add|reinforce|revise|ignore>] [limit=<n>]"
    filters: dict[str, object] = {
        "thread_id": None,
        "action": None,
        "limit": None,
    }
    for token in query.split():
        lowered = token.lower()
        if lowered.startswith("thread="):
            value = token.split("=", 1)[1].strip()
            if not value:
                return {}, usage
            filters["thread_id"] = value
            continue
        if lowered.startswith("limit="):
            value = token.split("=", 1)[1].strip()
            if not value.isdigit():
                return {}, usage
            limit = int(value)
            if limit <= 0:
                return {}, usage
            filters["limit"] = limit
            continue
        if lowered.startswith("action="):
            value = token.split("=", 1)[1].strip().lower()
            if value not in {"add", "reinforce", "revise", "ignore"}:
                return {}, usage
            filters["action"] = value
            continue
        if "=" in token:
            return {}, usage
        return {}, usage
    return {key: value for key, value in filters.items() if value is not None}, None


def filter_memories(
    memories: list[MemoryItem],
    *,
    category: str | None = None,
    importance: int | None = None,
    status: str | None = None,
) -> list[MemoryItem]:
    filtered = memories
    if category is not None:
        filtered = [item for item in filtered if item.category.lower() == category.lower()]
    if importance is not None:
        filtered = [item for item in filtered if item.importance == importance]
    if status is not None:
        filtered = [item for item in filtered if item.status.lower() == status.lower()]
    return filtered


def main() -> None:
    config = AgentConfig.from_env()
    runtime = build_runtime(config)
    agent = runtime.agent
    cli_store = runtime.cli_store
    user_id = config.user_id

    print(f"Agent ready for user '{user_id}' with backend '{config.backend}'. Type /help for commands, /exit to quit.")
    try:
        while True:
            user_text = input("you> ").strip()
            if user_text in {"/exit", "/quit"}:
                break
            if user_text == "/help":
                print(
                    "Commands: /profile, /memories [category=<name>] [importance=<1-5>] "
                    "[status=<active|superseded>] [query], /forget <memory_id>, "
                    "/dedupe-memories [thread=<thread_id>], /dedupe-log [thread=<thread_id>] [limit=<n>], "
                    "/memory-log [thread=<thread_id>] [action=<add|reinforce|revise|ignore>] [limit=<n>], "
                    "/learning, /routing, /thread <thread_id>, /exit"
                )
                continue
            if user_text == "/profile":
                print(format_profile(cli_store.get_profile()))
                continue
            if user_text.startswith("/memories"):
                raw_query = user_text.removeprefix("/memories").strip()
                query, filters, error = parse_memory_query(raw_query)
                if error:
                    print(error)
                    continue
                memories = (
                    cli_store.search_memories(query, limit=10, user_id=user_id)
                    if query
                    else cli_store.recent_memories(limit=10, user_id=user_id)
                )
                memories = filter_memories(memories, **filters)
                print(format_memories(memories))
                continue
            if user_text == "/learning":
                print(format_learning_events(cli_store.recent_learning_events(user_id=user_id, limit=10)))
                continue
            if user_text.startswith("/dedupe-log"):
                raw_query = user_text.removeprefix("/dedupe-log").strip()
                filters, error = parse_dedupe_log_query(raw_query)
                if error:
                    print(error)
                    continue
                print(
                    format_dedupe_events(
                        cli_store.recent_dedupe_events(
                            user_id=user_id,
                            limit=int(filters.get("limit", 10)),
                            thread_id=filters.get("thread_id"),
                        )
                    )
                )
                continue
            if user_text.startswith("/memory-log"):
                raw_query = user_text.removeprefix("/memory-log").strip()
                filters, error = parse_memory_evolution_log_query(raw_query)
                if error:
                    print(error)
                    continue
                print(
                    format_memory_evolution_events(
                        cli_store.recent_memory_evolution_events(
                            user_id=user_id,
                            limit=int(filters.get("limit", 10)),
                            thread_id=filters.get("thread_id"),
                            action=filters.get("action"),
                        )
                    )
                )
                continue
            if user_text.startswith("/thread "):
                thread_id = user_text.removeprefix("/thread").strip()
                if not thread_id:
                    print("Usage: /thread <thread_id>")
                    continue
                if isinstance(agent, ThreadInspectionRuntime):
                    inspection = agent.inspect_thread(thread_id, user_id=user_id)
                else:
                    inspection = ThreadInspection(
                        thread_id=thread_id,
                        transcript_messages=cli_store.thread_messages(thread_id, limit=50),
                        routing_events=cli_store.recent_routing_events(user_id=user_id, limit=20, thread_id=thread_id),
                        retrieval_events=cli_store.recent_retrieval_events(user_id=user_id, limit=20, thread_id=thread_id),
                        learning_events=cli_store.recent_learning_events(user_id=user_id, limit=20, thread_id=thread_id),
                        dedupe_events=cli_store.recent_dedupe_events(user_id=user_id, limit=20, thread_id=thread_id),
                        memory_evolution_events=cli_store.recent_memory_evolution_events(
                            user_id=user_id, limit=20, thread_id=thread_id
                        ),
                    )
                print(format_thread_inspection(inspection))
                continue
            if user_text == "/routing":
                print(format_routing_events(cli_store.recent_routing_events(user_id=user_id, limit=10)))
                continue
            if user_text.startswith("/routing "):
                query = user_text.removeprefix("/routing").strip()
                filters = parse_routing_query(query)
                events = cli_store.recent_routing_events(user_id=user_id, limit=50, **filters)
                print(format_routing_events(events))
                continue
            if user_text.startswith("/forget"):
                raw_id = user_text.removeprefix("/forget").strip()
                if not raw_id.isdigit():
                    print("Usage: /forget <memory_id>")
                    continue
                deleted = cli_store.delete_memory(int(raw_id), user_id=user_id)
                print("Memory deleted." if deleted else "No matching memory for current user.")
                continue
            if user_text.startswith("/dedupe-memories"):
                raw_query = user_text.removeprefix("/dedupe-memories").strip()
                filters, error = parse_dedupe_query(raw_query)
                if error:
                    print(error)
                    continue
                removed = cli_store.dedupe_memories(user_id=user_id, thread_id=filters.get("thread_id"))
                if removed.removed_count == 0:
                    print("No duplicate memories found.")
                    continue
                removed_labels = ", ".join(f"#{memory_id}" for memory_id in removed.removed_ids)
                kept_labels = ", ".join(f"#{memory_id}" for memory_id in removed.kept_ids)
                noun = "duplicate memory" if removed.removed_count == 1 else "duplicate memories"
                print(f"Removed {removed.removed_count} {noun}: {removed_labels}. Kept: {kept_labels}.")
                continue
            if not user_text:
                continue
            answer = agent.reply(user_text, user_id=user_id)
            print(f"agent> {answer}")
    finally:
        agent.close()


def format_profile(profile: AgentProfile) -> str:
    return "\n".join(
        [
            "Agent profile:",
            f"Identity: {profile.identity}",
            f"Style: {profile.style_notes}",
            f"Boundaries: {profile.boundaries}",
            f"Updated: {profile.updated_at}",
        ]
    )


def format_memories(memories: list[MemoryItem]) -> str:
    if not memories:
        return "No matching memories."
    return "\n".join(f"- #{item.id} [{item.category}/{item.importance}] [{item.status}] {item.content}" for item in memories)


def format_learning_events(events: list[LearningEvent]) -> str:
    if not events:
        return "No learning events."
    lines = []
    for event in events:
        fields = ",".join(event.profile_fields) if event.profile_fields else "none"
        preview = event.user_text.replace("\n", " ")[:60]
        lines.append(
            f"- [{event.created_at}] thread={event.thread_id} memories={event.memory_count} "
            f"profile={fields} user={preview}"
        )
    return "\n".join(lines)


def format_dedupe_events(events: list[DedupeEvent]) -> str:
    if not events:
        return "No dedupe events."
    lines = []
    for event in events:
        removed_ids = ",".join(str(memory_id) for memory_id in event.removed_ids) or "none"
        kept_ids = ",".join(str(memory_id) for memory_id in event.kept_ids) or "none"
        lines.append(f"- [{event.created_at}] removed={event.removed_count} ids={removed_ids} kept={kept_ids}")
    return "\n".join(lines)


def format_memory_evolution_events(events: list[MemoryEvolutionEvent]) -> str:
    if not events:
        return "No memory evolution events."
    lines = []
    for event in events:
        target_memory_id = event.target_memory_id if event.target_memory_id is not None else "none"
        result_memory_id = event.result_memory_id if event.result_memory_id is not None else "none"
        lines.append(
            f"- [{event.created_at}] action={event.action} target={target_memory_id} "
            f"result={result_memory_id} reason={event.reason}"
        )
    return "\n".join(lines)


def format_routing_events(events: list[RoutingEvent]) -> str:
    if not events:
        return "No routing events."
    lines = []
    for event in events:
        preview = event.user_text.replace("\n", " ")[:60]
        lines.append(
            f"- [{event.created_at}] thread={event.thread_id} "
            f"retrieve={event.should_retrieve}({event.retrieve_reason}) "
            f"learn={event.should_learn}({event.learn_reason}) user={preview}"
        )
    return "\n".join(lines)


def filter_routing_events(events: list[RoutingEvent], query: str) -> list[RoutingEvent]:
    query = query.strip()
    if not query:
        return events

    filtered = events
    for token in query.split():
        lowered = token.lower()
        if lowered.startswith("thread="):
            value = token.split("=", 1)[1]
            filtered = [event for event in filtered if event.thread_id == value]
            continue
        if lowered.startswith("learn="):
            value = lowered.split("=", 1)[1]
            want = value == "true"
            filtered = [event for event in filtered if event.should_learn is want]
            continue
        if lowered.startswith("retrieve="):
            value = lowered.split("=", 1)[1]
            want = value == "true"
            filtered = [event for event in filtered if event.should_retrieve is want]
            continue
        filtered = [
            event
            for event in filtered
            if lowered in event.retrieve_reason.lower()
            or lowered in event.learn_reason.lower()
            or lowered in event.user_text.lower()
        ]
    return filtered


def parse_routing_query(query: str) -> dict[str, object]:
    filters: dict[str, object] = {
        "thread_id": None,
        "learn": None,
        "retrieve": None,
        "reason": None,
        "text_query": None,
    }
    text_terms: list[str] = []
    for token in query.split():
        lowered = token.lower()
        if lowered.startswith("thread="):
            filters["thread_id"] = token.split("=", 1)[1]
            continue
        if lowered.startswith("learn="):
            filters["learn"] = lowered.split("=", 1)[1] == "true"
            continue
        if lowered.startswith("retrieve="):
            filters["retrieve"] = lowered.split("=", 1)[1] == "true"
            continue
        if token in {"low_signal", "recall_turn", "default_retrieve", "default_learn", "sensitive", "command", "empty", "too_short"}:
            filters["reason"] = token
            continue
        text_terms.append(token)
    if text_terms:
        filters["text_query"] = " ".join(text_terms)
    return filters
