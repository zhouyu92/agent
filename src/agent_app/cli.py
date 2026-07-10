from __future__ import annotations
from datetime import date, timedelta

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
    ReflectionEvent,
    RetrievalEvent,
    RoutingEvent,
    ThreadMessage,
)
from .runtime_agent import ConversationRuntime, ReflectionRuntime, ThreadInspection, ThreadInspectionRuntime
from .store import SqliteCliStore


_MEMORY_PREVIEW_CONTENT_LIMIT = 44


def build_agent(config: AgentConfig) -> tuple[ConversationRuntime, MemoryStore | SqliteCliStore]:
    runtime = build_runtime(config)
    return runtime.agent, runtime.cli_store


def parse_memory_query(query: str) -> tuple[str, dict[str, object], str | None]:
    usage = (
        "Usage: /memories [category=<name>] [importance=<1-5>] [status=<active|superseded|archived>] "
        "[confirmed_before=<YYYY-MM-DD>] [stale=<true|false>] [query]"
    )
    filters: dict[str, object] = {
        "category": None,
        "importance": None,
        "status": None,
        "confirmed_before": None,
        "stale": None,
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
            if value not in {"active", "superseded", "archived"}:
                return "", {}, usage
            filters["status"] = value
            continue
        if lowered.startswith("confirmed_before="):
            value = token.split("=", 1)[1].strip()
            try:
                date.fromisoformat(value)
            except ValueError:
                return "", {}, usage
            filters["confirmed_before"] = value
            continue
        if lowered.startswith("stale="):
            value = token.split("=", 1)[1].strip().lower()
            if value not in {"true", "false"}:
                return "", {}, usage
            filters["stale"] = value == "true"
            continue
        if "=" in token:
            return "", {}, usage
        text_terms.append(token)
    active_filters = {key: value for key, value in filters.items() if value is not None}
    return " ".join(text_terms), active_filters, None


def parse_memory_hygiene_query(query: str) -> tuple[dict[str, object], str | None]:
    usage = "Usage: /memory-hygiene [category=<name>] [days=<n>] [limit=<n>] [dry_run=<true|false>]"
    filters: dict[str, object] = {"category": None, "days": None, "limit": None, "dry_run": None}
    for token in query.split():
        lowered = token.lower()
        if lowered.startswith("category="):
            value = token.split("=", 1)[1].strip().lower()
            if not value:
                return {}, usage
            filters["category"] = value
            continue
        if lowered.startswith("days="):
            value = token.split("=", 1)[1].strip()
            if not value.isdigit():
                return {}, usage
            days = int(value)
            if days <= 0:
                return {}, usage
            filters["days"] = days
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
        if lowered.startswith("dry_run="):
            value = token.split("=", 1)[1].strip().lower()
            if value not in {"true", "false"}:
                return {}, usage
            filters["dry_run"] = value == "true"
            continue
        if "=" in token:
            return {}, usage
        return {}, usage
    return {key: value for key, value in filters.items() if value is not None}, None


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
    filters, text_terms, error = _parse_common_filter_tokens(query, usage=usage, allow_limit=True)
    if error:
        return {}, error
    if text_terms:
        return {}, usage
    return filters, None


def parse_memory_evolution_log_query(query: str) -> tuple[dict[str, object], str | None]:
    usage = "Usage: /memory-log [thread=<thread_id>] [action=<add|reinforce|revise|ignore|restore>] [reason=<name>] [limit=<n>]"
    filters, text_terms, error = _parse_common_filter_tokens(
        query,
        usage=usage,
        allow_reason=True,
        allow_limit=True,
        ignored_prefixes=("action=",),
    )
    if error:
        return {}, error
    if text_terms:
        return {}, usage
    filters["action"] = None
    for token in query.split():
        lowered = token.lower()
        if lowered.startswith("action="):
            value = token.split("=", 1)[1].strip().lower()
            if value not in {"add", "reinforce", "revise", "ignore", "restore"}:
                return {}, usage
            filters["action"] = value
            continue
    return {key: value for key, value in filters.items() if value is not None}, None


def parse_learning_query(query: str) -> tuple[dict[str, object], str | None]:
    usage = "Usage: /learning [thread=<thread_id>] [outcome=<memory+profile|memory_only|profile_only|no_change>] [limit=<n>]"
    filters, text_terms, error = _parse_common_filter_tokens(
        query,
        usage=usage,
        allow_limit=True,
        ignored_prefixes=("outcome=",),
    )
    if error:
        return {}, error
    if text_terms:
        return {}, usage
    filters["outcome"] = None
    for token in query.split():
        lowered = token.lower()
        if lowered.startswith("outcome="):
            value = token.split("=", 1)[1].strip().lower()
            if value not in {"memory+profile", "memory_only", "profile_only", "no_change"}:
                return {}, usage
            filters["outcome"] = value
            continue
    return {key: value for key, value in filters.items() if value is not None}, None


def parse_reflection_query(query: str) -> tuple[dict[str, object], str | None]:
    usage = "Usage: /reflections [thread=<thread_id>] [limit=<n>]"
    filters, text_terms, error = _parse_common_filter_tokens(query, usage=usage, allow_limit=True)
    if error or text_terms:
        return {}, error or usage
    return filters, None


def _parse_common_filter_tokens(
    query: str,
    *,
    usage: str,
    allow_reason: bool = False,
    allow_limit: bool = False,
    ignored_prefixes: tuple[str, ...] = (),
) -> tuple[dict[str, object], list[str], str | None]:
    filters: dict[str, object] = {"thread_id": None}
    if allow_reason:
        filters["reason"] = None
    if allow_limit:
        filters["limit"] = None

    text_terms: list[str] = []
    for token in query.split():
        lowered = token.lower()
        if any(lowered.startswith(prefix) for prefix in ignored_prefixes):
            continue
        if lowered.startswith("thread="):
            value = token.split("=", 1)[1].strip()
            if not value:
                return {}, [], usage
            filters["thread_id"] = value
            continue
        if allow_reason and lowered.startswith("reason="):
            value = token.split("=", 1)[1].strip()
            if not value:
                return {}, [], usage
            filters["reason"] = value
            continue
        if allow_limit and lowered.startswith("limit="):
            value = token.split("=", 1)[1].strip()
            if not value.isdigit():
                return {}, [], usage
            limit = int(value)
            if limit <= 0:
                return {}, [], usage
            filters["limit"] = limit
            continue
        if "=" in token:
            return {}, [], usage
        text_terms.append(token)
    return {key: value for key, value in filters.items() if value is not None}, text_terms, None


def filter_memories(
    memories: list[MemoryItem],
    *,
    category: str | None = None,
    importance: int | None = None,
    status: str | None = None,
    confirmed_before: str | None = None,
    stale: bool | None = None,
) -> list[MemoryItem]:
    filtered = memories
    if category is not None:
        filtered = [item for item in filtered if item.category.lower() == category.lower()]
    if importance is not None:
        filtered = [item for item in filtered if item.importance == importance]
    if status is not None:
        filtered = [item for item in filtered if item.status.lower() == status.lower()]
    if confirmed_before is not None:
        filtered = [
            item
            for item in filtered
            if item.last_confirmed_at is not None and item.last_confirmed_at[:10] < confirmed_before
        ]
    if stale:
        stale_cutoff = (_today_date() - timedelta(days=30)).isoformat()
        filtered = [
            item
            for item in filtered
            if item.last_confirmed_at is not None and item.last_confirmed_at[:10] < stale_cutoff
        ]
    return filtered


def _today_date() -> date:
    return date.today()


def filter_learning_events(events: list[LearningEvent], *, outcome: str | None = None) -> list[LearningEvent]:
    if outcome is None:
        return events
    return [event for event in events if _learning_outcome_label(event) == outcome]


def stale_memories(
    memories: list[MemoryItem],
    *,
    days: int,
    category: str | None = None,
) -> list[tuple[MemoryItem, int]]:
    cutoff = (_today_date() - timedelta(days=days)).isoformat()
    filtered = filter_memories(memories, category=category)
    stale_items: list[tuple[MemoryItem, int]] = []
    for item in filtered:
        if item.last_confirmed_at is None or item.last_confirmed_at[:10] >= cutoff:
            continue
        confirmed_date = date.fromisoformat(item.last_confirmed_at[:10])
        age_days = (_today_date() - confirmed_date).days
        stale_items.append((item, age_days))
    stale_items.sort(key=lambda pair: (-pair[1], -pair[0].importance, pair[0].id))
    return stale_items


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
                    "[status=<active|superseded|archived>] [confirmed_before=<YYYY-MM-DD>] [stale=<true|false>] [query], /forget <memory_id>, "
                    "/confirm-memory <memory_id>, /restore-memory <memory_id>, "
                    "/memory-hygiene [category=<name>] [days=<n>] [limit=<n>] [dry_run=<true|false>], "
                    "/confirm-stale [category=<name>] [days=<n>] [limit=<n>] [dry_run=<true|false>], "
                    "/forget-stale [category=<name>] [days=<n>] [limit=<n>] [dry_run=<true|false>], "
                    "/archive-stale [category=<name>] [days=<n>] [limit=<n>] [dry_run=<true|false>], "
                    "/dedupe-memories [thread=<thread_id>], /dedupe-log [thread=<thread_id>] [limit=<n>], "
                    "/memory-log [thread=<thread_id>] [action=<add|reinforce|revise|ignore|restore>] [reason=<name>] [limit=<n>], "
                    "/learning [thread=<thread_id>] [outcome=<memory+profile|memory_only|profile_only|no_change>] [limit=<n>], "
                    "/reflections [thread=<thread_id>] [limit=<n>], "
                    "/routing [thread=<thread_id>] [learn=<true|false>] [retrieve=<true|false>] [reason=<name>] [limit=<n>] [text], "
                    "/reflect [thread_id], /summarize <thread_id>, /thread <thread_id>, /exit"
                )
                continue
            if user_text == "/profile":
                print(format_profile(cli_store.get_profile()))
                continue
            if user_text.startswith("/memory-hygiene"):
                raw_query = user_text.removeprefix("/memory-hygiene").strip()
                filters, error = parse_memory_hygiene_query(raw_query)
                if error:
                    print(error)
                    continue
                output_limit = int(filters.get("limit", 10))
                candidate_limit = max(output_limit, 100)
                items = stale_memories(
                    cli_store.recent_memories(limit=candidate_limit, user_id=user_id),
                    days=int(filters.get("days", 30)),
                    category=filters.get("category"),
                )[:output_limit]
                print(format_memory_hygiene(items, days=int(filters.get("days", 30))))
                continue
            if user_text.startswith("/confirm-stale"):
                raw_query = user_text.removeprefix("/confirm-stale").strip()
                filters, error = parse_memory_hygiene_query(raw_query)
                if error:
                    print("Usage: /confirm-stale [category=<name>] [days=<n>] [limit=<n>] [dry_run=<true|false>]")
                    continue
                output_limit = int(filters.get("limit", 10))
                candidate_limit = max(output_limit, 100)
                items = stale_memories(
                    cli_store.recent_memories(limit=candidate_limit, user_id=user_id),
                    days=int(filters.get("days", 30)),
                    category=filters.get("category"),
                )[:output_limit]
                if not items:
                    print("No stale memories matched.")
                    continue
                if filters.get("dry_run") is True:
                    print(format_stale_memory_preview("confirm", items))
                    continue
                confirmed_ids = [
                    item.id
                    for item, _ in items
                    if cli_store.confirm_memory(item.id, user_id=user_id)
                ]
                if not confirmed_ids:
                    print("No stale memories matched.")
                    continue
                noun = "stale memory" if len(confirmed_ids) == 1 else "stale memories"
                labels = ", ".join(f"#{memory_id}" for memory_id in confirmed_ids)
                print(f"Confirmed {len(confirmed_ids)} {noun}: {labels}.")
                continue
            if user_text.startswith("/forget-stale"):
                raw_query = user_text.removeprefix("/forget-stale").strip()
                filters, error = parse_memory_hygiene_query(raw_query)
                if error:
                    print("Usage: /forget-stale [category=<name>] [days=<n>] [limit=<n>] [dry_run=<true|false>]")
                    continue
                output_limit = int(filters.get("limit", 10))
                candidate_limit = max(output_limit, 100)
                items = stale_memories(
                    cli_store.recent_memories(limit=candidate_limit, user_id=user_id),
                    days=int(filters.get("days", 30)),
                    category=filters.get("category"),
                )[:output_limit]
                if not items:
                    print("No stale memories matched.")
                    continue
                if filters.get("dry_run") is True:
                    print(format_stale_memory_preview("forget", items))
                    continue
                deleted_ids = [
                    item.id
                    for item, _ in items
                    if cli_store.delete_memory(item.id, user_id=user_id)
                ]
                if not deleted_ids:
                    print("No stale memories matched.")
                    continue
                noun = "stale memory" if len(deleted_ids) == 1 else "stale memories"
                labels = ", ".join(f"#{memory_id}" for memory_id in deleted_ids)
                print(f"Forgot {len(deleted_ids)} {noun}: {labels}.")
                continue
            if user_text.startswith("/archive-stale"):
                raw_query = user_text.removeprefix("/archive-stale").strip()
                filters, error = parse_memory_hygiene_query(raw_query)
                if error:
                    print("Usage: /archive-stale [category=<name>] [days=<n>] [limit=<n>] [dry_run=<true|false>]")
                    continue
                output_limit = int(filters.get("limit", 10))
                candidate_limit = max(output_limit, 100)
                items = stale_memories(
                    cli_store.recent_memories(limit=candidate_limit, user_id=user_id, status="active"),
                    days=int(filters.get("days", 30)),
                    category=filters.get("category"),
                )[:output_limit]
                if not items:
                    print("No stale memories matched.")
                    continue
                if filters.get("dry_run") is True:
                    print(format_stale_memory_preview("archive", items))
                    continue
                archived_ids = [
                    item.id
                    for item, _ in items
                    if cli_store.archive_memory(item.id, user_id=user_id)
                ]
                if not archived_ids:
                    print("No stale memories matched.")
                    continue
                noun = "stale memory" if len(archived_ids) == 1 else "stale memories"
                labels = ", ".join(f"#{memory_id}" for memory_id in archived_ids)
                print(f"Archived {len(archived_ids)} {noun}: {labels}.")
                continue
            if user_text.startswith("/memories"):
                raw_query = user_text.removeprefix("/memories").strip()
                query, filters, error = parse_memory_query(raw_query)
                if error:
                    print(error)
                    continue
                status = str(filters.get("status") or "active")
                memories = (
                    (
                        cli_store.search_memories(query, limit=10, user_id=user_id)
                        if status == "active"
                        else cli_store.search_memories(query, limit=10, user_id=user_id, status=status)
                    )
                    if query
                    else (
                        cli_store.recent_memories(limit=10, user_id=user_id)
                        if status == "active"
                        else cli_store.recent_memories(limit=10, user_id=user_id, status=status)
                    )
                )
                memories = filter_memories(memories, **filters)
                print(format_memories(memories))
                continue
            if user_text.startswith("/learning"):
                raw_query = user_text.removeprefix("/learning").strip()
                filters, error = parse_learning_query(raw_query)
                if error:
                    print(error)
                    continue
                events = cli_store.recent_learning_events(
                    user_id=user_id,
                    limit=int(filters.get("limit", 10)),
                    thread_id=filters.get("thread_id"),
                )
                print(format_learning_events(filter_learning_events(events, outcome=filters.get("outcome"))))
                continue
            if user_text.startswith("/reflections"):
                raw_query = user_text.removeprefix("/reflections").strip()
                filters, error = parse_reflection_query(raw_query)
                if error:
                    print(error)
                    continue
                events = cli_store.recent_reflection_events(
                    user_id=user_id,
                    limit=int(filters.get("limit", 10)),
                    thread_id=filters.get("thread_id"),
                )
                print(format_reflection_events(events))
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
                            reason=filters.get("reason"),
                        )
                    )
                )
                continue
            if user_text == "/reflect" or user_text.startswith("/reflect "):
                thread_id = user_text.removeprefix("/reflect").strip() or "default"
                if not isinstance(agent, ReflectionRuntime):
                    print("Reflection is not available for this runtime.")
                    continue
                result = agent.reflect(thread_id=thread_id, user_id=user_id)
                if result.status == "not_ready":
                    print("Reflection needs at least two unreviewed learning events in this thread.")
                    continue
                if result.status == "invalid_output":
                    print("Reflection produced invalid output; no changes were saved.")
                    continue
                print(
                    f"Reflection completed for {len(result.source_event_ids)} episodes: "
                    f"{result.memory_count} memory updates, {len(result.profile_fields or [])} profile updates."
                )
                continue
            if user_text.startswith("/summarize"):
                thread_id = user_text.removeprefix("/summarize").strip()
                summarize = getattr(agent, "summarize_thread", None)
                if not thread_id:
                    print("Usage: /summarize <thread_id>")
                elif summarize is None:
                    print("Thread summary is not available for this runtime.")
                else:
                    summary = summarize(thread_id, user_id=user_id)
                    print(summary or "No thread messages to summarize.")
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
                        reflection_events=cli_store.recent_reflection_events(user_id=user_id, limit=20, thread_id=thread_id),
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
                filters, error = parse_routing_query(query)
                if error:
                    print(error)
                    continue
                events = cli_store.recent_routing_events(user_id=user_id, limit=int(filters.pop("limit", 50)), **filters)
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
            if user_text.startswith("/confirm-memory"):
                raw_id = user_text.removeprefix("/confirm-memory").strip()
                if not raw_id.isdigit():
                    print("Usage: /confirm-memory <memory_id>")
                    continue
                confirmed = cli_store.confirm_memory(int(raw_id), user_id=user_id)
                print("Memory confirmed." if confirmed else "No matching active memory for current user.")
                continue
            if user_text.startswith("/restore-memory"):
                raw_id = user_text.removeprefix("/restore-memory").strip()
                if not raw_id.isdigit():
                    print("Usage: /restore-memory <memory_id>")
                    continue
                restored = cli_store.restore_memory(int(raw_id), user_id=user_id)
                print("Memory restored." if restored else "No matching archived memory for current user.")
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
    return "\n".join(
        f"- #{item.id} [{item.category}/{item.importance}] [{item.status}] "
        f"confirmed={item.last_confirmed_at or 'none'} {_memory_content_preview(item.content)}"
        for item in memories
    )


def format_memory_hygiene(items: list[tuple[MemoryItem, int]], *, days: int) -> str:
    if not items:
        return f"Memory hygiene (stale>{days}d):\nNo stale memories."
    lines = [f"Memory hygiene (stale>{days}d):"]
    lines.extend(
        f"- #{item.id} [{item.category}/{item.importance}] age={age_days}d "
        f"confirmed={item.last_confirmed_at or 'none'} {_memory_content_preview(item.content)}"
        for item, age_days in items
    )
    return "\n".join(lines)


def format_stale_memory_preview(action: str, items: list[tuple[MemoryItem, int]]) -> str:
    noun = "stale memory" if len(items) == 1 else "stale memories"
    lines = [f"Dry run: would {action} {len(items)} {noun}:"]
    lines.extend(
        f"- #{item.id} [{item.category}/{item.importance}] age={age_days}d "
        f"confirmed={item.last_confirmed_at or 'none'} {_memory_content_preview(item.content)}"
        for item, age_days in items
    )
    return "\n".join(lines)


def _memory_content_preview(content: str) -> str:
    if len(content) <= _MEMORY_PREVIEW_CONTENT_LIMIT:
        return content
    return f"{content[:_MEMORY_PREVIEW_CONTENT_LIMIT]}..."


def format_learning_events(events: list[LearningEvent]) -> str:
    if not events:
        return "No learning events."
    lines = []
    for event in events:
        fields = ",".join(event.profile_fields) if event.profile_fields else "none"
        outcome = _learning_outcome_label(event)
        preview = event.user_text.replace("\n", " ")[:60]
        lines.append(
            f"- [{event.created_at}] thread={event.thread_id} memories={event.memory_count} outcome={outcome} "
            f"profile={fields} user={preview}"
        )
    return "\n".join(lines)


def format_reflection_events(events: list[ReflectionEvent]) -> str:
    if not events:
        return "No reflection events."
    lines = []
    for event in events:
        source_ids = ",".join(str(event_id) for event_id in event.source_event_ids) or "none"
        fields = ",".join(event.profile_fields) if event.profile_fields else "none"
        summary = event.summary.replace("\n", " ")[:80]
        lines.append(
            f"- [{event.created_at}] thread={event.thread_id} episodes={source_ids} "
            f"memories={event.memory_count} profile={fields} summary={summary}"
        )
    return "\n".join(lines)


def _learning_outcome_label(event: LearningEvent) -> str:
    has_memories = event.memory_count > 0
    has_profile_updates = bool(event.profile_fields)
    if has_memories and has_profile_updates:
        return "memory+profile"
    if has_memories:
        return "memory_only"
    if has_profile_updates:
        return "profile_only"
    return "no_change"


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
        candidate_preview = event.candidate_content.replace("\n", " ")[:80]
        lines.append(
            f"- [{event.created_at}] action={event.action} category={event.candidate_category} "
            f"target={target_memory_id} result={result_memory_id} reason={event.reason} "
            f"candidate={candidate_preview}"
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


def parse_routing_query(query: str) -> tuple[dict[str, object], str | None]:
    usage = "Usage: /routing [thread=<thread_id>] [learn=<true|false>] [retrieve=<true|false>] [reason=<name>] [limit=<n>] [text]"
    filters, text_terms, error = _parse_common_filter_tokens(
        query,
        usage=usage,
        allow_reason=True,
        allow_limit=True,
        ignored_prefixes=("learn=", "retrieve="),
    )
    if error:
        return {}, error
    filters["learn"] = None
    filters["retrieve"] = None
    filters["text_query"] = None
    text_terms: list[str] = []
    for token in query.split():
        lowered = token.lower()
        if lowered.startswith("thread=") or lowered.startswith("reason=") or lowered.startswith("limit="):
            continue
        if lowered.startswith("learn="):
            value = lowered.split("=", 1)[1].strip()
            if value not in {"true", "false"}:
                return {}, usage
            filters["learn"] = value == "true"
            continue
        if lowered.startswith("retrieve="):
            value = lowered.split("=", 1)[1].strip()
            if value not in {"true", "false"}:
                return {}, usage
            filters["retrieve"] = value == "true"
            continue
        if token in {"low_signal", "recall_turn", "default_retrieve", "default_learn", "sensitive", "command", "empty", "too_short"}:
            filters["reason"] = token
            continue
        if "=" in token:
            return {}, usage
        text_terms.append(token)
    if text_terms:
        filters["text_query"] = " ".join(text_terms)
    return filters, None
