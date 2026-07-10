from datetime import date

from agent_app.cli import (
    _parse_common_filter_tokens,
    build_agent,
    filter_routing_events,
    format_audit_timeline,
    format_checkpoint_diff,
    format_retrieval_comparison,
    format_thread_inspection,
    format_memory_evolution_events,
    format_learning_events,
    format_reflection_events,
    main,
    parse_learning_query,
    parse_reflection_query,
    parse_memory_evolution_log_query,
    parse_dedupe_log_query,
    parse_dedupe_query,
    parse_memory_query,
    format_memories,
    format_memory_hygiene,
    format_profile,
    format_routing_events,
    format_checkpoint_messages,
    parse_memory_hygiene_query,
)
from agent_app.agent import ConversationalAgent
from agent_app.bootstrap import build_runtime
from agent_app.config import AgentConfig
from agent_app.langgraph_agent import LangGraphAgent
from agent_app.memory import AgentProfile, DedupeEvent, DedupeResult, LearningEvent, MemoryEvolutionEvent, MemoryItem, ReflectionEvent, RetrievalEvent, RoutingEvent, ThreadMessage
from agent_app.reflection import ReflectionRunResult
from agent_app.runtime_agent import ConversationRuntime, ThreadInspection, ThreadInspectionRuntime
from agent_app.store import (
    SqliteAuditStore,
    SqliteCliStore,
    SqliteLongTermStore,
    SqliteProfileStore,
    SqliteSemanticMemoryStore,
    SqliteTranscriptStore,
)
from agent_app.thread_inspection import LangGraphThreadInspectionBuilder
from agent_app.thread_state import LangGraphCheckpointStateReader, LangGraphThreadStateStore


def test_main_runs_dedupe_memories_command(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def __init__(self):
            self.calls = []

        def dedupe_memories(self, user_id: str, thread_id: str | None = None) -> DedupeResult:
            self.calls.append((user_id, thread_id))
            return DedupeResult(removed_count=2, removed_ids=[7, 9], kept_ids=[3])

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    fake_store = FakeCliStore()
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(fake_store))
    inputs = iter(["/dedupe-memories", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert fake_store.calls == [("alice", None)]
    assert "Removed 2 duplicate memories: #7, #9. Kept: #3." in output


def test_main_runs_dedupe_memories_command_with_thread_filter(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def __init__(self):
            self.calls = []

        def dedupe_memories(self, user_id: str, thread_id: str | None = None) -> DedupeResult:
            self.calls.append((user_id, thread_id))
            return DedupeResult(removed_count=1, removed_ids=[7], kept_ids=[3])

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    fake_store = FakeCliStore()
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(fake_store))
    inputs = iter(["/dedupe-memories thread=t10", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert fake_store.calls == [("alice", "t10")]
    assert "Removed 1 duplicate memory: #7. Kept: #3." in output


def test_main_uses_singular_dedupe_message_for_one_memory(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def dedupe_memories(self, user_id: str, thread_id: str | None = None) -> DedupeResult:
            return DedupeResult(removed_count=1, removed_ids=[7], kept_ids=[3])

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(FakeCliStore()))
    inputs = iter(["/dedupe-memories", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert "Removed 1 duplicate memory: #7. Kept: #3." in output


def test_main_rejects_invalid_dedupe_memories_filter(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def dedupe_memories(self, user_id: str, thread_id: str | None = None) -> DedupeResult:
            raise AssertionError("dedupe_memories should not be called for invalid filters")

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(FakeCliStore()))
    inputs = iter(["/dedupe-memories source=conversation", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert "Usage: /dedupe-memories [thread=<thread_id>]" in output


def test_main_help_shows_memory_filter_syntax(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        pass

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(FakeCliStore()))
    inputs = iter(["/help", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert (
        "/memories [category=<name>] [importance=<1-5>] [status=<active|superseded|archived>] "
        "[confirmed_before=<YYYY-MM-DD>] [stale=<true|false>] [query]"
    ) in output
    assert "/dedupe-memories [thread=<thread_id>]" in output
    assert "/dedupe-log [thread=<thread_id>] [limit=<n>]" in output
    assert "/memory-log [thread=<thread_id>] [action=<add|reinforce|revise|ignore|restore>] [reason=<name>] [limit=<n>]" in output
    assert "/learning [thread=<thread_id>] [outcome=<memory+profile|memory_only|profile_only|no_change>] [limit=<n>]" in output
    assert "/routing [thread=<thread_id>] [learn=<true|false>] [retrieve=<true|false>] [reason=<name>] [limit=<n>] [text]" in output
    assert "/memory-hygiene [category=<name>] [days=<n>] [limit=<n>]" in output
    assert "/confirm-memory <memory_id>" in output
    assert "/confirm-stale [category=<name>] [days=<n>] [limit=<n>]" in output
    assert "/forget-stale [category=<name>] [days=<n>] [limit=<n>]" in output
    assert "/archive-stale [category=<name>] [days=<n>] [limit=<n>]" in output
    assert "/reflect [thread_id]" in output
    assert "dry_run=<true|false>" in output


def test_main_confirms_memory(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def __init__(self):
            self.calls = []

        def confirm_memory(self, memory_id: int, user_id: str):
            self.calls.append((memory_id, user_id))
            return True

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    fake_store = FakeCliStore()
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(fake_store))
    inputs = iter(["/confirm-memory 7", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert fake_store.calls == [(7, "alice")]
    assert "Memory confirmed." in output


def test_main_rejects_invalid_confirm_memory_usage(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def confirm_memory(self, memory_id: int, user_id: str):
            raise AssertionError("confirm_memory should not be called for invalid usage")

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(FakeCliStore()))
    inputs = iter(["/confirm-memory abc", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert "Usage: /confirm-memory <memory_id>" in output


def test_main_restores_archived_memory(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def __init__(self):
            self.calls = []

        def restore_memory(self, memory_id: int, user_id: str):
            self.calls.append((memory_id, user_id))
            return True

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    fake_store = FakeCliStore()
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )
    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(fake_store))
    inputs = iter(["/restore-memory 7", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert fake_store.calls == [(7, "alice")]
    assert "Memory restored." in output


def test_main_runs_reflection_for_requested_thread(monkeypatch, capsys):
    class FakeAgent:
        def __init__(self):
            self.calls = []

        def reflect(self, thread_id: str, user_id: str):
            self.calls.append((thread_id, user_id))
            return ReflectionRunResult(status="completed", source_event_ids=[1, 2], memory_count=1, profile_fields=["style_notes"])

        def close(self):
            return None

    class FakeRuntime:
        def __init__(self, agent):
            self.agent = agent
            self.cli_store = object()

    agent = FakeAgent()
    config = AgentConfig(api_key="test-key", base_url="https://example.test/compatible-mode/v1", backend="classic", user_id="alice")
    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(agent))
    inputs = iter(["/reflect t1", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert agent.calls == [("t1", "alice")]
    assert "Reflection completed for 2 episodes: 1 memory updates, 1 profile updates." in output


def test_main_rejects_invalid_restore_memory_usage(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def restore_memory(self, memory_id: int, user_id: str):
            raise AssertionError("restore_memory should not be called for invalid usage")

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )
    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(FakeCliStore()))
    inputs = iter(["/restore-memory abc", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert "Usage: /restore-memory <memory_id>" in output


def test_main_confirms_stale_memories(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def __init__(self):
            self.recent_calls = []
            self.confirm_calls = []

        def recent_memories(self, limit: int, user_id: str, status: str = "active"):
            self.recent_calls.append((limit, user_id))
            return [
                MemoryItem(
                    id=1,
                    category="preference",
                    content="用户喜欢先给结论。",
                    importance=4,
                    source="conversation",
                    created_at="2026-06-01T00:00:00+00:00",
                    status="active",
                    last_confirmed_at="2026-06-05T00:00:00+00:00",
                ),
                MemoryItem(
                    id=2,
                    category="preference",
                    content="用户喜欢先给结论再补原因。",
                    importance=3,
                    source="conversation",
                    created_at="2026-06-02T00:00:00+00:00",
                    status="active",
                    last_confirmed_at="2026-06-10T00:00:00+00:00",
                ),
                MemoryItem(
                    id=3,
                    category="fact",
                    content="用户正在搭建 agent。",
                    importance=4,
                    source="conversation",
                    created_at="2026-07-08T00:00:00+00:00",
                    status="active",
                    last_confirmed_at="2026-07-08T00:00:00+00:00",
                ),
            ]

        def confirm_memory(self, memory_id: int, user_id: str):
            self.confirm_calls.append((memory_id, user_id))
            return True

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    fake_store = FakeCliStore()
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(fake_store))
    monkeypatch.setattr("agent_app.cli._today_date", lambda: date(2026, 7, 9))
    inputs = iter(["/confirm-stale category=preference days=25 limit=1", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert fake_store.recent_calls == [(100, "alice")]
    assert fake_store.confirm_calls == [(1, "alice")]
    assert "Confirmed 1 stale memory: #1." in output


def test_main_reports_when_no_stale_memories_match_confirm_stale(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def __init__(self):
            self.confirm_calls = []

        def recent_memories(self, limit: int, user_id: str):
            return [
                MemoryItem(
                    id=3,
                    category="fact",
                    content="用户正在搭建 agent。",
                    importance=4,
                    source="conversation",
                    created_at="2026-07-08T00:00:00+00:00",
                    status="active",
                    last_confirmed_at="2026-07-08T00:00:00+00:00",
                ),
            ]

        def confirm_memory(self, memory_id: int, user_id: str):
            self.confirm_calls.append((memory_id, user_id))
            return True

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    fake_store = FakeCliStore()
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(fake_store))
    monkeypatch.setattr("agent_app.cli._today_date", lambda: date(2026, 7, 9))
    inputs = iter(["/confirm-stale category=preference days=25 limit=5", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert fake_store.confirm_calls == []
    assert "No stale memories matched." in output


def test_main_rejects_invalid_confirm_stale_usage(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def recent_memories(self, limit: int, user_id: str):
            raise AssertionError("recent_memories should not be called for invalid usage")

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(FakeCliStore()))
    inputs = iter(["/confirm-stale days=zero", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert "Usage: /confirm-stale [category=<name>] [days=<n>] [limit=<n>] [dry_run=<true|false>]" in output


def test_main_previews_confirm_stale_without_mutation(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def __init__(self):
            self.confirm_calls = []

        def recent_memories(self, limit: int, user_id: str):
            return [
                MemoryItem(
                    id=1,
                    category="preference",
                    content="用户喜欢先给结论。",
                    importance=4,
                    source="conversation",
                    created_at="2026-06-01T00:00:00+00:00",
                    status="active",
                    last_confirmed_at="2026-06-05T00:00:00+00:00",
                ),
            ]

        def confirm_memory(self, memory_id: int, user_id: str):
            self.confirm_calls.append((memory_id, user_id))
            return True

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    fake_store = FakeCliStore()
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(fake_store))
    monkeypatch.setattr("agent_app.cli._today_date", lambda: date(2026, 7, 9))
    inputs = iter(["/confirm-stale days=25 limit=1 dry_run=true", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert fake_store.confirm_calls == []
    assert "Dry run: would confirm 1 stale memory:" in output
    assert "#1 [preference/4]" in output
    assert "age=34d" in output
    assert "confirmed=2026-06-05T00:00:00+00:00" in output
    assert "用户喜欢先给结论。" in output


def test_main_previews_confirm_stale_truncates_long_content(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def __init__(self):
            self.confirm_calls = []

        def recent_memories(self, limit: int, user_id: str):
            return [
                MemoryItem(
                    id=1,
                    category="preference",
                    content="用户希望每次回复都先给非常简洁的结论，然后再补充必要背景和关键判断依据，尤其是在复杂问题里也不要省略结论。",
                    importance=4,
                    source="conversation",
                    created_at="2026-06-01T00:00:00+00:00",
                    status="active",
                    last_confirmed_at="2026-06-05T00:00:00+00:00",
                ),
            ]

        def confirm_memory(self, memory_id: int, user_id: str):
            self.confirm_calls.append((memory_id, user_id))
            return True

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    fake_store = FakeCliStore()
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(fake_store))
    monkeypatch.setattr("agent_app.cli._today_date", lambda: date(2026, 7, 9))
    inputs = iter(["/confirm-stale days=25 limit=1 dry_run=true", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert fake_store.confirm_calls == []
    assert "用户希望每次回复都先给非常简洁的结论，然后再补充必要背景和关键判断依据，尤其是在复杂问题..." in output
    assert "不要省略结论。" not in output


def test_main_forgets_stale_memories(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def __init__(self):
            self.recent_calls = []
            self.delete_calls = []

        def recent_memories(self, limit: int, user_id: str, status: str = "active"):
            self.recent_calls.append((limit, user_id))
            return [
                MemoryItem(
                    id=1,
                    category="preference",
                    content="用户喜欢先给结论。",
                    importance=4,
                    source="conversation",
                    created_at="2026-06-01T00:00:00+00:00",
                    status="active",
                    last_confirmed_at="2026-06-05T00:00:00+00:00",
                ),
                MemoryItem(
                    id=2,
                    category="preference",
                    content="用户喜欢先给结论再补原因。",
                    importance=3,
                    source="conversation",
                    created_at="2026-06-02T00:00:00+00:00",
                    status="active",
                    last_confirmed_at="2026-06-10T00:00:00+00:00",
                ),
                MemoryItem(
                    id=3,
                    category="fact",
                    content="用户正在搭建 agent。",
                    importance=4,
                    source="conversation",
                    created_at="2026-07-08T00:00:00+00:00",
                    status="active",
                    last_confirmed_at="2026-07-08T00:00:00+00:00",
                ),
            ]

        def delete_memory(self, memory_id: int, user_id: str):
            self.delete_calls.append((memory_id, user_id))
            return True

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    fake_store = FakeCliStore()
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(fake_store))
    monkeypatch.setattr("agent_app.cli._today_date", lambda: date(2026, 7, 9))
    inputs = iter(["/forget-stale category=preference days=25 limit=1", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert fake_store.recent_calls == [(100, "alice")]
    assert fake_store.delete_calls == [(1, "alice")]
    assert "Forgot 1 stale memory: #1." in output


def test_main_reports_when_no_stale_memories_match_forget_stale(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def __init__(self):
            self.delete_calls = []

        def recent_memories(self, limit: int, user_id: str, status: str = "active"):
            return [
                MemoryItem(
                    id=3,
                    category="fact",
                    content="用户正在搭建 agent。",
                    importance=4,
                    source="conversation",
                    created_at="2026-07-08T00:00:00+00:00",
                    status="active",
                    last_confirmed_at="2026-07-08T00:00:00+00:00",
                ),
            ]

        def delete_memory(self, memory_id: int, user_id: str):
            self.delete_calls.append((memory_id, user_id))
            return True

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    fake_store = FakeCliStore()
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(fake_store))
    monkeypatch.setattr("agent_app.cli._today_date", lambda: date(2026, 7, 9))
    inputs = iter(["/forget-stale category=preference days=25 limit=5", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert fake_store.delete_calls == []
    assert "No stale memories matched." in output


def test_main_rejects_invalid_forget_stale_usage(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def recent_memories(self, limit: int, user_id: str):
            raise AssertionError("recent_memories should not be called for invalid usage")

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(FakeCliStore()))
    inputs = iter(["/forget-stale days=zero", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert "Usage: /forget-stale [category=<name>] [days=<n>] [limit=<n>] [dry_run=<true|false>]" in output


def test_main_previews_forget_stale_without_mutation(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def __init__(self):
            self.delete_calls = []

        def recent_memories(self, limit: int, user_id: str):
            return [
                MemoryItem(
                    id=1,
                    category="preference",
                    content="用户喜欢先给结论。",
                    importance=4,
                    source="conversation",
                    created_at="2026-06-01T00:00:00+00:00",
                    status="active",
                    last_confirmed_at="2026-06-05T00:00:00+00:00",
                ),
            ]

        def delete_memory(self, memory_id: int, user_id: str):
            self.delete_calls.append((memory_id, user_id))
            return True

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    fake_store = FakeCliStore()
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(fake_store))
    monkeypatch.setattr("agent_app.cli._today_date", lambda: date(2026, 7, 9))
    inputs = iter(["/forget-stale days=25 limit=1 dry_run=true", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert fake_store.delete_calls == []
    assert "Dry run: would forget 1 stale memory:" in output
    assert "#1 [preference/4]" in output
    assert "age=34d" in output
    assert "confirmed=2026-06-05T00:00:00+00:00" in output
    assert "用户喜欢先给结论。" in output


def test_main_archives_stale_memories(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def __init__(self):
            self.recent_calls = []
            self.archive_calls = []

        def recent_memories(self, limit: int, user_id: str, status: str = "active"):
            self.recent_calls.append((limit, user_id, status))
            return [
                MemoryItem(
                    id=1,
                    category="preference",
                    content="用户喜欢先给结论。",
                    importance=4,
                    source="conversation",
                    created_at="2026-06-01T00:00:00+00:00",
                    status="active",
                    last_confirmed_at="2026-06-05T00:00:00+00:00",
                ),
                MemoryItem(
                    id=2,
                    category="fact",
                    content="用户正在搭建 agent。",
                    importance=4,
                    source="conversation",
                    created_at="2026-07-08T00:00:00+00:00",
                    status="active",
                    last_confirmed_at="2026-07-08T00:00:00+00:00",
                ),
            ]

        def archive_memory(self, memory_id: int, user_id: str):
            self.archive_calls.append((memory_id, user_id))
            return True

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    fake_store = FakeCliStore()
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(fake_store))
    monkeypatch.setattr("agent_app.cli._today_date", lambda: date(2026, 7, 9))
    inputs = iter(["/archive-stale category=preference days=25 limit=1", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert fake_store.recent_calls == [(100, "alice", "active")]
    assert fake_store.archive_calls == [(1, "alice")]
    assert "Archived 1 stale memory: #1." in output


def test_main_previews_archive_stale_without_mutation(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def __init__(self):
            self.archive_calls = []

        def recent_memories(self, limit: int, user_id: str, status: str = "active"):
            return [
                MemoryItem(
                    id=1,
                    category="preference",
                    content="用户喜欢先给结论。",
                    importance=4,
                    source="conversation",
                    created_at="2026-06-01T00:00:00+00:00",
                    status="active",
                    last_confirmed_at="2026-06-05T00:00:00+00:00",
                ),
            ]

        def archive_memory(self, memory_id: int, user_id: str):
            self.archive_calls.append((memory_id, user_id))
            return True

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    fake_store = FakeCliStore()
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(fake_store))
    monkeypatch.setattr("agent_app.cli._today_date", lambda: date(2026, 7, 9))
    inputs = iter(["/archive-stale days=25 limit=1 dry_run=true", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert fake_store.archive_calls == []
    assert "Dry run: would archive 1 stale memory:" in output
    assert "#1 [preference/4]" in output


def test_main_rejects_invalid_archive_stale_usage(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def recent_memories(self, limit: int, user_id: str, status: str = "active"):
            raise AssertionError("recent_memories should not be called for invalid usage")

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(FakeCliStore()))
    inputs = iter(["/archive-stale days=zero", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert "Usage: /archive-stale [category=<name>] [days=<n>] [limit=<n>] [dry_run=<true|false>]" in output


def test_format_memory_hygiene_truncates_long_content():
    text = format_memory_hygiene(
        [
            (
                MemoryItem(
                    id=1,
                    category="preference",
                    content="用户希望每次回复都先给非常简洁的结论，然后再补充必要背景和关键判断依据，尤其是在复杂问题里也不要省略结论。",
                    importance=4,
                    source="conversation",
                    created_at="2026-06-01T00:00:00+00:00",
                    status="active",
                    last_confirmed_at="2026-06-05T00:00:00+00:00",
                ),
                34,
            )
        ],
        days=30,
    )

    assert "用户希望每次回复都先给非常简洁的结论，然后再补充必要背景和关键判断依据，尤其是在复杂问题..." in text
    assert "不要省略结论。" not in text


def test_parse_memory_hygiene_query_extracts_filters():
    filters, error = parse_memory_hygiene_query("category=preference days=45 limit=5 dry_run=true")

    assert filters == {"category": "preference", "days": 45, "limit": 5, "dry_run": True}
    assert error is None


def test_parse_memory_hygiene_query_rejects_invalid_days():
    filters, error = parse_memory_hygiene_query("days=zero")

    assert filters == {}
    assert error == "Usage: /memory-hygiene [category=<name>] [days=<n>] [limit=<n>] [dry_run=<true|false>]"


def test_format_memory_hygiene_shows_empty_state():
    assert format_memory_hygiene([], days=30) == "Memory hygiene (stale>30d):\nNo stale memories."


def test_main_shows_memory_hygiene_view(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def __init__(self):
            self.calls = []

        def recent_memories(self, limit: int, user_id: str):
            self.calls.append((limit, user_id))
            return [
                MemoryItem(
                    id=1,
                    category="preference",
                    content="用户喜欢先给结论。",
                    importance=4,
                    source="conversation",
                    created_at="2026-06-01T00:00:00+00:00",
                    status="active",
                    last_confirmed_at="2026-06-05T00:00:00+00:00",
                ),
                MemoryItem(
                    id=2,
                    category="fact",
                    content="用户正在搭建 agent。",
                    importance=4,
                    source="conversation",
                    created_at="2026-07-08T00:00:00+00:00",
                    status="active",
                    last_confirmed_at="2026-07-08T00:00:00+00:00",
                ),
            ]

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    fake_store = FakeCliStore()
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(fake_store))
    monkeypatch.setattr("agent_app.cli._today_date", lambda: date(2026, 7, 9))
    inputs = iter(["/memory-hygiene category=preference days=30 limit=5", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert fake_store.calls == [(100, "alice")]
    assert "Memory hygiene (stale>30d):" in output
    assert "#1" in output
    assert "age=34d" in output
    assert "#2" not in output


def test_parse_learning_query_extracts_thread_and_outcome_filters():
    filters, error = parse_learning_query("thread=t10 outcome=no_change limit=5")

    assert filters == {"thread_id": "t10", "outcome": "no_change", "limit": 5}
    assert error is None


def test_parse_learning_query_rejects_unknown_outcome():
    filters, error = parse_learning_query("outcome=archived")

    assert filters == {}
    assert error == (
        "Usage: /learning [thread=<thread_id>] "
        "[outcome=<memory+profile|memory_only|profile_only|no_change>] [limit=<n>]"
    )


def test_parse_learning_query_rejects_invalid_limit():
    filters, error = parse_learning_query("limit=zero")

    assert filters == {}
    assert error == (
        "Usage: /learning [thread=<thread_id>] "
        "[outcome=<memory+profile|memory_only|profile_only|no_change>] [limit=<n>]"
    )


def test_main_shows_learning_events_with_filters(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def __init__(self):
            self.calls = []

        def recent_learning_events(self, user_id: str, limit: int, *, thread_id: str | None = None):
            self.calls.append((user_id, limit, thread_id))
            return [
                LearningEvent(
                    id=1,
                    user_id=user_id,
                    thread_id="t10",
                    user_text="继续。",
                    assistant_text="好的。",
                    memory_count=0,
                    profile_fields=[],
                    created_at="2026-07-02T00:02:00+00:00",
                ),
                LearningEvent(
                    id=2,
                    user_id=user_id,
                    thread_id="t10",
                    user_text="以后回答先给结论。",
                    assistant_text="我记住了。",
                    memory_count=1,
                    profile_fields=["style_notes"],
                    created_at="2026-07-02T00:00:00+00:00",
                ),
            ]

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    fake_store = FakeCliStore()
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(fake_store))
    inputs = iter(["/learning thread=t10 outcome=no_change limit=5", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert fake_store.calls == [("alice", 5, "t10")]
    assert "outcome=no_change" in output
    assert "outcome=memory+profile" not in output


def test_main_shows_recent_memory_evolution_events(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def recent_memory_evolution_events(
            self,
            user_id: str,
            limit: int,
            thread_id: str | None = None,
            action: str | None = None,
            reason: str | None = None,
        ):
            return [
                MemoryEvolutionEvent(
                    id=1,
                    user_id=user_id,
                    thread_id=thread_id,
                    action="revise",
                    candidate_category="preference",
                    candidate_content="以后回答先给结论。",
                    target_memory_id=3,
                    result_memory_id=4,
                    reason="correction_phrase",
                    created_at="2026-07-04T12:00:00+00:00",
                )
            ]

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(FakeCliStore()))
    inputs = iter(["/memory-log", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert "2026-07-04T12:00:00+00:00" in output
    assert "action=revise" in output
    assert "target=3" in output
    assert "result=4" in output


def test_main_shows_recent_memory_evolution_events_with_thread_filter(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def __init__(self):
            self.calls = []

        def recent_memory_evolution_events(
            self,
            user_id: str,
            limit: int,
            thread_id: str | None = None,
            action: str | None = None,
            reason: str | None = None,
        ):
            self.calls.append((user_id, limit, thread_id, action, reason))
            return [
                MemoryEvolutionEvent(
                    id=1,
                    user_id=user_id,
                    thread_id=thread_id,
                    action="reinforce",
                    candidate_category="preference",
                    candidate_content="用户还是喜欢先给结论。",
                    target_memory_id=3,
                    result_memory_id=3,
                    reason="confirmed_existing_memory",
                    created_at="2026-07-04T12:00:00+00:00",
                )
            ]

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    fake_store = FakeCliStore()
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(fake_store))
    inputs = iter(["/memory-log thread=t10", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert fake_store.calls == [("alice", 10, "t10", None, None)]
    assert "action=reinforce" in output


def test_main_shows_recent_memory_evolution_events_with_action_filter(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def __init__(self):
            self.calls = []

        def recent_memory_evolution_events(
            self,
            user_id: str,
            limit: int,
            thread_id: str | None = None,
            action: str | None = None,
            reason: str | None = None,
        ):
            self.calls.append((user_id, limit, thread_id, action, reason))
            return [
                MemoryEvolutionEvent(
                    id=1,
                    user_id=user_id,
                    thread_id=thread_id,
                    action="revise",
                    candidate_category="preference",
                    candidate_content="以后回答先给结论。",
                    target_memory_id=3,
                    result_memory_id=4,
                    reason="correction_phrase",
                    created_at="2026-07-04T12:00:00+00:00",
                )
            ]

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    fake_store = FakeCliStore()
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(fake_store))
    inputs = iter(["/memory-log action=revise", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert fake_store.calls == [("alice", 10, None, "revise", None)]
    assert "action=revise" in output


def test_main_rejects_invalid_memory_log_filter(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def recent_memory_evolution_events(
            self,
            user_id: str,
            limit: int,
            thread_id: str | None = None,
            action: str | None = None,
            reason: str | None = None,
        ):
            raise AssertionError("recent_memory_evolution_events should not be called for invalid filters")

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(FakeCliStore()))
    inputs = iter(["/memory-log source=conversation", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert "Usage: /memory-log [thread=<thread_id>] [action=<add|reinforce|revise|ignore|restore>] [reason=<name>] [limit=<n>]" in output


def test_main_shows_recent_dedupe_events(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def recent_dedupe_events(self, user_id: str, limit: int, thread_id: str | None = None):
            return [
                DedupeEvent(
                    id=1,
                    user_id=user_id,
                    thread_id=None,
                    removed_count=2,
                    removed_ids=[7, 9],
                    kept_ids=[3],
                    created_at="2026-07-04T12:00:00+00:00",
                )
            ]

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(FakeCliStore()))
    inputs = iter(["/dedupe-log", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert "2026-07-04T12:00:00+00:00" in output
    assert "removed=2" in output
    assert "ids=7,9" in output
    assert "kept=3" in output


def test_main_shows_recent_dedupe_events_with_thread_filter(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def __init__(self):
            self.calls = []

        def recent_dedupe_events(self, user_id: str, limit: int, thread_id: str | None = None):
            self.calls.append((user_id, limit, thread_id))
            return [
                DedupeEvent(
                    id=1,
                    user_id=user_id,
                    thread_id=thread_id,
                    removed_count=1,
                    removed_ids=[7],
                    kept_ids=[3],
                    created_at="2026-07-04T12:00:00+00:00",
                )
            ]

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    fake_store = FakeCliStore()
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(fake_store))
    inputs = iter(["/dedupe-log thread=t10", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert fake_store.calls == [("alice", 10, "t10")]
    assert "ids=7" in output
    assert "kept=3" in output


def test_main_shows_recent_dedupe_events_with_limit_filter(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def __init__(self):
            self.calls = []

        def recent_dedupe_events(self, user_id: str, limit: int, thread_id: str | None = None):
            self.calls.append((user_id, limit, thread_id))
            return [
                DedupeEvent(
                    id=1,
                    user_id=user_id,
                    thread_id=None,
                    removed_count=1,
                    removed_ids=[7],
                    kept_ids=[3],
                    created_at="2026-07-04T12:00:00+00:00",
                )
            ]

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    fake_store = FakeCliStore()
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(fake_store))
    inputs = iter(["/dedupe-log limit=5", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert fake_store.calls == [("alice", 5, None)]
    assert "ids=7" in output
    assert "kept=3" in output


def test_main_rejects_invalid_dedupe_log_filter(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def recent_dedupe_events(self, user_id: str, limit: int, thread_id: str | None = None):
            raise AssertionError("recent_dedupe_events should not be called for invalid filters")

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(FakeCliStore()))
    inputs = iter(["/dedupe-log source=conversation", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert "Usage: /dedupe-log [thread=<thread_id>]" in output


def test_main_rejects_invalid_dedupe_log_limit(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def recent_dedupe_events(self, user_id: str, limit: int, thread_id: str | None = None):
            raise AssertionError("recent_dedupe_events should not be called for invalid filters")

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(FakeCliStore()))
    inputs = iter(["/dedupe-log limit=abc", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert "Usage: /dedupe-log [thread=<thread_id>] [limit=<n>]" in output


def test_format_dedupe_events_shows_empty_state():
    from agent_app.cli import format_dedupe_events

    assert format_dedupe_events([]) == "No dedupe events."


def test_format_memory_evolution_events_shows_empty_state():
    assert format_memory_evolution_events([]) == "No memory evolution events."


def test_format_memory_evolution_events_shows_category_and_candidate_preview():
    events = [
        MemoryEvolutionEvent(
            id=1,
            user_id="alice",
            thread_id="t10",
            action="ignore",
            candidate_category="preference",
            candidate_content="用户偏好回答时先给结论，再补充原因。",
            target_memory_id=3,
            result_memory_id=None,
            reason="no_new_information",
            created_at="2026-07-04T12:00:00+00:00",
        )
    ]

    text = format_memory_evolution_events(events)

    assert "action=ignore" in text
    assert "category=preference" in text
    assert "candidate=用户偏好回答时先给结论，再补充原因。" in text
    assert "reason=no_new_information" in text


def test_main_filters_memories_by_category_and_importance(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def recent_memories(self, limit: int, user_id: str, status: str = "active"):
            return [
                MemoryItem(
                    id=1,
                    category="preference",
                    content="用户喜欢先给结论。",
                    importance=4,
                    source="conversation",
                    created_at="2026-07-02T00:00:00+00:00",
                ),
                MemoryItem(
                    id=2,
                    category="fact",
                    content="用户正在搭建 agent。",
                    importance=4,
                    source="conversation",
                    created_at="2026-07-02T00:01:00+00:00",
                ),
                MemoryItem(
                    id=3,
                    category="preference",
                    content="用户喜欢简洁回答。",
                    importance=2,
                    source="conversation",
                    created_at="2026-07-02T00:02:00+00:00",
                ),
            ]

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(FakeCliStore()))
    inputs = iter(["/memories category=preference importance=4", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert "#1" in output
    assert "用户喜欢先给结论。" in output
    assert "#2" not in output
    assert "#3" not in output


def test_main_filters_memories_by_status(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def recent_memories(self, limit: int, user_id: str, status: str = "active"):
            return [
                MemoryItem(
                    id=1,
                    category="preference",
                    content="用户喜欢先给结论。",
                    importance=4,
                    source="conversation",
                    created_at="2026-07-02T00:00:00+00:00",
                    status="active",
                ),
                MemoryItem(
                    id=2,
                    category="preference",
                    content="用户以前喜欢先铺垫。",
                    importance=3,
                    source="conversation",
                    created_at="2026-07-02T00:01:00+00:00",
                    status="superseded",
                ),
            ]

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(FakeCliStore()))
    inputs = iter(["/memories status=superseded", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert "#1" not in output
    assert "#2" in output
    assert "[superseded]" in output
    assert "用户以前喜欢先铺垫。" in output


def test_main_filters_memories_by_confirmed_before(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def recent_memories(self, limit: int, user_id: str):
            return [
                MemoryItem(
                    id=1,
                    category="preference",
                    content="用户喜欢先给结论。",
                    importance=4,
                    source="conversation",
                    created_at="2026-07-02T00:00:00+00:00",
                    status="active",
                    last_confirmed_at="2026-07-03T00:00:00+00:00",
                ),
                MemoryItem(
                    id=2,
                    category="fact",
                    content="用户正在搭建 agent。",
                    importance=4,
                    source="conversation",
                    created_at="2026-07-02T00:01:00+00:00",
                    status="active",
                    last_confirmed_at="2026-07-08T00:00:00+00:00",
                ),
            ]

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(FakeCliStore()))
    inputs = iter(["/memories confirmed_before=2026-07-05", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert "#1" in output
    assert "#2" not in output


def test_main_filters_memories_by_stale(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def recent_memories(self, limit: int, user_id: str):
            return [
                MemoryItem(
                    id=1,
                    category="preference",
                    content="用户喜欢先给结论。",
                    importance=4,
                    source="conversation",
                    created_at="2026-06-01T00:00:00+00:00",
                    status="active",
                    last_confirmed_at="2026-06-05T00:00:00+00:00",
                ),
                MemoryItem(
                    id=2,
                    category="fact",
                    content="用户正在搭建 agent。",
                    importance=4,
                    source="conversation",
                    created_at="2026-07-01T00:00:00+00:00",
                    status="active",
                    last_confirmed_at="2026-07-08T00:00:00+00:00",
                ),
            ]

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(FakeCliStore()))
    monkeypatch.setattr("agent_app.cli._today_date", lambda: date(2026, 7, 9))
    inputs = iter(["/memories stale=true", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert "#1" in output
    assert "#2" not in output


def test_main_rejects_invalid_memory_importance_filter(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def recent_memories(self, limit: int, user_id: str):
            raise AssertionError("recent_memories should not be called for invalid filters")

        def search_memories(self, query: str, limit: int, user_id: str):
            raise AssertionError("search_memories should not be called for invalid filters")

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(FakeCliStore()))
    inputs = iter(["/memories importance=abc", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert (
        "Usage: /memories [category=<name>] [importance=<1-5>] [status=<active|superseded|archived>] "
        "[confirmed_before=<YYYY-MM-DD>] [stale=<true|false>] [query]"
    ) in output


def test_main_rejects_empty_memory_category_filter(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def recent_memories(self, limit: int, user_id: str):
            raise AssertionError("recent_memories should not be called for invalid filters")

        def search_memories(self, query: str, limit: int, user_id: str):
            raise AssertionError("search_memories should not be called for invalid filters")

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(FakeCliStore()))
    inputs = iter(["/memories category=", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert (
        "Usage: /memories [category=<name>] [importance=<1-5>] [status=<active|superseded|archived>] "
        "[confirmed_before=<YYYY-MM-DD>] [stale=<true|false>] [query]"
    ) in output


def test_main_rejects_unknown_memory_filter(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def recent_memories(self, limit: int, user_id: str):
            raise AssertionError("recent_memories should not be called for invalid filters")

        def search_memories(self, query: str, limit: int, user_id: str):
            raise AssertionError("search_memories should not be called for invalid filters")

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(FakeCliStore()))
    inputs = iter(["/memories source=conversation", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert (
        "Usage: /memories [category=<name>] [importance=<1-5>] [status=<active|superseded|archived>] "
        "[confirmed_before=<YYYY-MM-DD>] [stale=<true|false>] [query]"
    ) in output


def test_parse_memory_query_extracts_filters_and_text_query():
    query, filters, error = parse_memory_query(
        "category=preference importance=4 status=active confirmed_before=2026-07-05 stale=true 结论优先"
    )

    assert query == "结论优先"
    assert filters == {
        "category": "preference",
        "importance": 4,
        "status": "active",
        "confirmed_before": "2026-07-05",
        "stale": True,
    }
    assert error is None


def test_parse_memory_query_rejects_unknown_status():
    query, filters, error = parse_memory_query("status=paused")

    assert query == ""
    assert filters == {}
    assert error == (
        "Usage: /memories [category=<name>] [importance=<1-5>] [status=<active|superseded|archived>] "
        "[confirmed_before=<YYYY-MM-DD>] [stale=<true|false>] [query]"
    )


def test_parse_memory_query_accepts_archived_status():
    query, filters, error = parse_memory_query("status=archived")

    assert query == ""
    assert filters == {"status": "archived"}
    assert error is None


def test_parse_memory_query_rejects_invalid_confirmed_before():
    query, filters, error = parse_memory_query("confirmed_before=soon")

    assert query == ""
    assert filters == {}
    assert error == (
        "Usage: /memories [category=<name>] [importance=<1-5>] [status=<active|superseded|archived>] "
        "[confirmed_before=<YYYY-MM-DD>] [stale=<true|false>] [query]"
    )


def test_parse_memory_query_rejects_invalid_stale_value():
    query, filters, error = parse_memory_query("stale=maybe")

    assert query == ""
    assert filters == {}
    assert error == (
        "Usage: /memories [category=<name>] [importance=<1-5>] [status=<active|superseded|archived>] "
        "[confirmed_before=<YYYY-MM-DD>] [stale=<true|false>] [query]"
    )


def test_parse_memory_evolution_log_query_extracts_filters():
    filters, error = parse_memory_evolution_log_query("thread=t10 action=revise reason=correction_phrase limit=5")

    assert filters == {
        "thread_id": "t10",
        "action": "revise",
        "reason": "correction_phrase",
        "limit": 5,
    }
    assert error is None


def test_parse_memory_evolution_log_query_rejects_unknown_action():
    filters, error = parse_memory_evolution_log_query("action=archive")

    assert filters == {}
    assert error == "Usage: /memory-log [thread=<thread_id>] [action=<add|reinforce|revise|ignore|restore>] [reason=<name>] [limit=<n>]"


def test_parse_memory_evolution_log_query_accepts_restore_action():
    filters, error = parse_memory_evolution_log_query("action=restore")

    assert error is None
    assert filters == {"action": "restore"}


def test_main_shows_recent_memory_evolution_events_with_reason_filter(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def __init__(self):
            self.calls = []

        def recent_memory_evolution_events(
            self,
            user_id: str,
            limit: int,
            thread_id: str | None = None,
            action: str | None = None,
            reason: str | None = None,
        ):
            self.calls.append((user_id, limit, thread_id, action, reason))
            return [
                MemoryEvolutionEvent(
                    id=1,
                    user_id=user_id,
                    thread_id=thread_id,
                    action="ignore",
                    candidate_category="preference",
                    candidate_content="用户偏好回答时先给结论，再补充原因。",
                    target_memory_id=3,
                    result_memory_id=None,
                    reason=reason or "no_new_information",
                    created_at="2026-07-04T12:00:00+00:00",
                )
            ]

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    fake_store = FakeCliStore()
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(fake_store))
    inputs = iter(["/memory-log reason=no_new_information", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert fake_store.calls == [("alice", 10, None, None, "no_new_information")]
    assert "reason=no_new_information" in output


def test_parse_memory_query_rejects_unknown_filter():
    query, filters, error = parse_memory_query("source=conversation")

    assert query == ""
    assert filters == {}
    assert error == (
        "Usage: /memories [category=<name>] [importance=<1-5>] [status=<active|superseded|archived>] "
        "[confirmed_before=<YYYY-MM-DD>] [stale=<true|false>] [query]"
    )


def test_parse_dedupe_query_extracts_thread_filter():
    filters, error = parse_dedupe_query("thread=t10")

    assert filters == {"thread_id": "t10"}
    assert error is None


def test_parse_dedupe_log_query_extracts_thread_filter():
    filters, error = parse_dedupe_log_query("thread=t10")

    assert filters == {"thread_id": "t10"}
    assert error is None


def test_parse_dedupe_log_query_extracts_limit_filter():
    filters, error = parse_dedupe_log_query("limit=5")

    assert filters == {"limit": 5}
    assert error is None


def test_format_profile_shows_current_agent_profile():
    profile = AgentProfile(
        identity="长期陪伴 agent",
        style_notes="先给结论",
        boundaries="不保存敏感信息",
        updated_at="2026-07-02T00:00:00+00:00",
    )

    text = format_profile(profile)

    assert "长期陪伴 agent" in text
    assert "先给结论" in text
    assert "不保存敏感信息" in text


def test_format_memories_shows_empty_state():
    assert format_memories([]) == "No matching memories."


def test_format_memories_lists_memory_items():
    memories = [
        MemoryItem(
            id=1,
            category="preference",
            content="用户喜欢先给结论。",
            importance=4,
            source="conversation",
            created_at="2026-07-02T00:00:00+00:00",
            status="active",
            last_confirmed_at="2026-07-03T00:00:00+00:00",
        )
    ]

    text = format_memories(memories)

    assert "#1" in text
    assert "[preference/4]" in text
    assert "[active]" in text
    assert "confirmed=2026-07-03T00:00:00+00:00" in text
    assert "用户喜欢先给结论。" in text


def test_format_learning_events_lists_recent_events():
    events = [
        LearningEvent(
            id=1,
            user_id="alice",
            thread_id="t1",
            user_text="以后回答先给结论。",
            assistant_text="我记住了。",
            memory_count=1,
            profile_fields=["style_notes"],
            created_at="2026-07-02T00:00:00+00:00",
        )
    ]

    text = format_learning_events(events)

    assert "t1" in text
    assert "memories=1" in text
    assert "outcome=memory+profile" in text
    assert "style_notes" in text


def test_format_learning_events_marks_no_change_outcome():
    events = [
        LearningEvent(
            id=1,
            user_id="alice",
            thread_id="t2",
            user_text="继续。",
            assistant_text="好的。",
            memory_count=0,
            profile_fields=[],
            created_at="2026-07-02T00:02:00+00:00",
        )
    ]

    text = format_learning_events(events)

    assert "t2" in text
    assert "memories=0" in text
    assert "outcome=no_change" in text
    assert "profile=none" in text


def test_format_reflection_events_lists_episode_ids_and_summary():
    text = format_reflection_events(
        [
            ReflectionEvent(
                id=1,
                user_id="alice",
                thread_id="t1",
                source_event_ids=[3, 4],
                summary="用户稳定偏好先给结论。",
                memory_count=1,
                profile_fields=["style_notes"],
                created_at="2026-07-10T00:00:00+00:00",
            )
        ]
    )

    assert "thread=t1" in text
    assert "episodes=3,4" in text
    assert "memories=1" in text
    assert "用户稳定偏好先给结论。" in text


def test_parse_reflection_query_accepts_thread_and_limit():
    filters, error = parse_reflection_query("thread=t1 limit=3")

    assert error is None
    assert filters == {"thread_id": "t1", "limit": 3}


def test_format_routing_events_shows_empty_state():
    assert format_routing_events([]) == "No routing events."


def test_format_routing_events_lists_recent_events():
    events = [
        RoutingEvent(
            id=1,
            user_id="alice",
            thread_id="t10",
            user_text="谢谢",
            should_retrieve=False,
            retrieve_reason="low_signal",
            should_learn=False,
            learn_reason="low_signal",
            created_at="2026-07-02T00:00:00+00:00",
        )
    ]

    text = format_routing_events(events)

    assert "t10" in text
    assert "retrieve=False(low_signal)" in text
    assert "learn=False(low_signal)" in text


def test_filter_routing_events_by_thread_and_learn_flag():
    events = [
        RoutingEvent(
            id=1,
            user_id="alice",
            thread_id="t10",
            user_text="谢谢",
            should_retrieve=False,
            retrieve_reason="low_signal",
            should_learn=False,
            learn_reason="low_signal",
            created_at="2026-07-02T00:00:00+00:00",
        ),
        RoutingEvent(
            id=2,
            user_id="alice",
            thread_id="t11",
            user_text="你还记得我的回答偏好吗？",
            should_retrieve=True,
            retrieve_reason="default_retrieve",
            should_learn=False,
            learn_reason="recall_turn",
            created_at="2026-07-02T00:01:00+00:00",
        ),
    ]

    assert [event.id for event in filter_routing_events(events, "thread=t11")] == [2]
    assert [event.id for event in filter_routing_events(events, "learn=false")] == [1, 2]


def test_filter_routing_events_by_reason_or_preview_text():
    events = [
        RoutingEvent(
            id=1,
            user_id="alice",
            thread_id="t10",
            user_text="谢谢",
            should_retrieve=False,
            retrieve_reason="low_signal",
            should_learn=False,
            learn_reason="low_signal",
            created_at="2026-07-02T00:00:00+00:00",
        ),
        RoutingEvent(
            id=2,
            user_id="alice",
            thread_id="t11",
            user_text="你还记得我的回答偏好吗？",
            should_retrieve=True,
            retrieve_reason="default_retrieve",
            should_learn=False,
            learn_reason="recall_turn",
            created_at="2026-07-02T00:01:00+00:00",
        ),
    ]

    assert [event.id for event in filter_routing_events(events, "recall_turn")] == [2]
    assert [event.id for event in filter_routing_events(events, "谢谢")] == [1]


def test_parse_routing_query_extracts_store_filters():
    from agent_app.cli import parse_routing_query

    filters, error = parse_routing_query("thread=t10 learn=false limit=5 recall_turn")

    assert filters == {
        "thread_id": "t10",
        "learn": False,
        "retrieve": None,
        "reason": "recall_turn",
        "limit": 5,
        "text_query": None,
    }
    assert error is None


def test_parse_common_filter_tokens_extracts_thread_reason_limit_and_text():
    filters, text_terms, error = _parse_common_filter_tokens(
        "thread=t10 reason=recall_turn limit=5 hello world",
        usage="Usage: test",
        allow_reason=True,
        allow_limit=True,
    )

    assert filters == {"thread_id": "t10", "reason": "recall_turn", "limit": 5}
    assert text_terms == ["hello", "world"]
    assert error is None


def test_parse_common_filter_tokens_rejects_invalid_limit():
    filters, text_terms, error = _parse_common_filter_tokens(
        "limit=zero",
        usage="Usage: test",
        allow_limit=True,
    )

    assert filters == {}
    assert text_terms == []
    assert error == "Usage: test"


def test_parse_routing_query_rejects_invalid_limit():
    from agent_app.cli import parse_routing_query

    filters, error = parse_routing_query("limit=zero")

    assert filters == {}
    assert error == (
        "Usage: /routing [thread=<thread_id>] [learn=<true|false>] "
        "[retrieve=<true|false>] [reason=<name>] [limit=<n>] [text]"
    )


def test_parse_routing_query_rejects_invalid_boolean():
    from agent_app.cli import parse_routing_query

    filters, error = parse_routing_query("learn=maybe")

    assert filters == {}
    assert error == (
        "Usage: /routing [thread=<thread_id>] [learn=<true|false>] "
        "[retrieve=<true|false>] [reason=<name>] [limit=<n>] [text]"
    )


def test_main_rejects_invalid_routing_filter(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def recent_routing_events(self, *args, **kwargs):
            raise AssertionError("recent_routing_events should not be called for invalid filters")

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(FakeCliStore()))
    inputs = iter(["/routing learn=maybe", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert (
        "Usage: /routing [thread=<thread_id>] [learn=<true|false>] "
        "[retrieve=<true|false>] [reason=<name>] [limit=<n>] [text]"
    ) in output


def test_main_shows_routing_events_with_limit_filter(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def __init__(self):
            self.calls = []

        def recent_routing_events(
            self,
            user_id: str,
            limit: int = 10,
            *,
            thread_id: str | None = None,
            learn: bool | None = None,
            retrieve: bool | None = None,
            reason: str | None = None,
            text_query: str | None = None,
        ):
            self.calls.append((user_id, limit, thread_id, learn, retrieve, reason, text_query))
            return [
                RoutingEvent(
                    id=1,
                    user_id=user_id,
                    thread_id="t10",
                    user_text="谢谢",
                    should_retrieve=False,
                    retrieve_reason="low_signal",
                    should_learn=False,
                    learn_reason="low_signal",
                    created_at="2026-07-02T00:00:00+00:00",
                )
            ]

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    fake_store = FakeCliStore()
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        backend="classic",
        user_id="alice",
    )

    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(fake_store))
    inputs = iter(["/routing limit=5 recall_turn", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert fake_store.calls == [("alice", 5, None, None, None, "recall_turn", None)]
    assert "retrieve=False" in output


def test_format_audit_timeline_combines_routing_and_learning_sections():
    messages = [
        ThreadMessage(
            id=1,
            thread_id="t10",
            role="user",
            content="谢谢",
            created_at="2026-07-02T00:00:00+00:00",
        ),
        ThreadMessage(
            id=2,
            thread_id="t10",
            role="assistant",
            content="不客气。",
            created_at="2026-07-02T00:00:30+00:00",
        ),
    ]
    routing_events = [
        RoutingEvent(
            id=1,
            user_id="alice",
            thread_id="t10",
            user_text="谢谢",
            should_retrieve=False,
            retrieve_reason="low_signal",
            should_learn=False,
            learn_reason="low_signal",
            created_at="2026-07-02T00:00:00+00:00",
        )
    ]
    retrieval_events = [
        RetrievalEvent(
            id=1,
            user_id="alice",
            thread_id="t10",
            user_text="谢谢",
            memory_count=1,
            memory_ids=[7],
            memory_preview="用户喜欢先给结论再补充原因。",
            created_at="2026-07-02T00:00:15+00:00",
        )
    ]
    learning_events = [
        LearningEvent(
            id=1,
            user_id="alice",
            thread_id="t10",
            user_text="以后回答先给结论。",
            assistant_text="我记住了。",
            memory_count=1,
            profile_fields=["style_notes"],
            created_at="2026-07-02T00:01:00+00:00",
        )
    ]
    dedupe_events = [
        DedupeEvent(
            id=1,
            user_id="alice",
            thread_id="t10",
            removed_count=1,
            removed_ids=[7],
            kept_ids=[3],
            created_at="2026-07-02T00:00:45+00:00",
        )
    ]

    text = format_audit_timeline("t10", messages, routing_events, retrieval_events, learning_events, dedupe_events)

    assert "Thread audit: t10" in text
    user_index = text.index("message user 谢谢")
    assistant_index = text.index("message assistant 不客气。")
    routing_index = text.index("routing retrieve=False(low_signal)")
    retrieval_index = text.index("retrieval memories=1")
    dedupe_index = text.index("dedupe removed=1 ids=7 kept=3")
    learning_index = text.index("learning memories=1")
    assert user_index < routing_index < retrieval_index < assistant_index < dedupe_index < learning_index
    assert "谢谢" in text
    assert "以后回答先给结论。" in text
    assert "先给结论" in text
    assert "ids=7" in text
    assert "kept=3" in text
    assert "outcome=memory+profile" in text


def test_format_audit_timeline_shows_memory_evolution_candidate_preview():
    text = format_audit_timeline(
        "t10",
        messages=[],
        routing_events=[],
        retrieval_events=[],
        learning_events=[],
        dedupe_events=[],
        memory_evolution_events=[
            MemoryEvolutionEvent(
                id=1,
                user_id="alice",
                thread_id="t10",
                action="ignore",
                candidate_category="preference",
                candidate_content="用户偏好回答时先给结论，再补充原因。",
                target_memory_id=3,
                result_memory_id=None,
                reason="no_new_information",
                created_at="2026-07-04T12:00:00+00:00",
            )
        ],
    )

    assert "memory_evolution ignore" in text
    assert "category=preference" in text
    assert "candidate=用户偏好回答时先给结论，再补充原因。" in text
    assert "reason=no_new_information" in text


def test_format_audit_timeline_shows_reflection_event():
    text = format_audit_timeline(
        "t10",
        messages=[],
        routing_events=[],
        retrieval_events=[],
        learning_events=[],
        dedupe_events=[],
        reflection_events=[
            ReflectionEvent(
                id=1,
                user_id="alice",
                thread_id="t10",
                source_event_ids=[1, 2],
                summary="用户稳定偏好先给结论。",
                memory_count=1,
                profile_fields=[],
                created_at="2026-07-10T00:00:00+00:00",
            )
        ],
    )

    assert "reflection episodes=1,2 memories=1 profile=none" in text


def test_main_lists_reflection_events(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def __init__(self):
            self.calls = []

        def recent_reflection_events(self, user_id: str, limit: int, thread_id: str | None = None):
            self.calls.append((user_id, limit, thread_id))
            return [
                ReflectionEvent(
                    id=1,
                    user_id=user_id,
                    thread_id="t1",
                    source_event_ids=[1, 2],
                    summary="用户稳定偏好先给结论。",
                    memory_count=1,
                    profile_fields=[],
                    created_at="2026-07-10T00:00:00+00:00",
                )
            ]

    class FakeRuntime:
        def __init__(self, cli_store):
            self.agent = FakeAgent()
            self.cli_store = cli_store

    store = FakeCliStore()
    config = AgentConfig(api_key="test-key", base_url="https://example.test/compatible-mode/v1", user_id="alice")
    monkeypatch.setattr("agent_app.cli.AgentConfig.from_env", lambda: config)
    monkeypatch.setattr("agent_app.cli.build_runtime", lambda _: FakeRuntime(store))
    inputs = iter(["/reflections thread=t1 limit=3", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))

    main()

    output = capsys.readouterr().out
    assert store.calls == [("alice", 3, "t1")]
    assert "episodes=1,2" in output


def test_format_checkpoint_messages_shows_checkpoint_state_separately():
    class Message:
        def __init__(self, type_, content):
            self.type = type_
            self.content = content

    text = format_checkpoint_messages(
        "t10",
        [
            Message("human", "谢谢"),
            Message("ai", "不客气。"),
        ],
    )

    assert "Checkpoint state: t10" in text
    assert "- human 谢谢" in text
    assert "- ai 不客气。" in text


def test_format_checkpoint_messages_marks_unavailable_checkpoint_runtime():
    text = format_checkpoint_messages("t10", [], checkpoint_available=False)

    assert text == "Checkpoint state: t10\nCheckpoint not available for this runtime."


def test_format_checkpoint_messages_marks_empty_checkpoint_state():
    text = format_checkpoint_messages("t10", [], checkpoint_available=True)

    assert text == "Checkpoint state: t10\nNo checkpoint messages."


def test_format_checkpoint_messages_shows_snapshot_metadata():
    text = format_checkpoint_messages(
        "t10",
        [],
        checkpoint_available=True,
        checkpoint_state_keys=["messages", "retrieved_memories", "routing_decision"],
        checkpoint_message_count=2,
        checkpoint_step=1,
        checkpoint_updated_at="2026-07-03T10:49:32.171604+00:00",
        checkpoint_routing_decision={
            "should_retrieve": False,
            "retrieve_reason": "too_short",
            "should_learn": False,
            "learn_reason": "too_short",
        },
        checkpoint_retrieved_memories=[],
    )

    assert "Checkpoint state: t10" in text
    assert "- keys messages, retrieved_memories, routing_decision" in text
    assert "- messages 2" in text
    assert "- step 1" in text
    assert "- updated_at 2026-07-03T10:49:32.171604+00:00" in text
    assert "- routing retrieve=False(too_short) learn=False(too_short)" in text
    assert "- retrieved_memories 0" in text
    assert "No checkpoint messages." in text


def test_format_checkpoint_messages_shows_retrieved_memory_preview():
    text = format_checkpoint_messages(
        "t10",
        [],
        checkpoint_available=True,
        checkpoint_retrieved_memories=[
            {
                "category": "preference",
                "content": "用户喜欢先给结论再补充原因。",
                "importance": 4,
                "source": "conversation",
            },
            {
                "category": "fact",
                "content": "用户正在搭建一个会持续学习的 agent。",
                "importance": 5,
                "source": "conversation",
            },
        ],
    )

    assert "- retrieved_memories 2" in text
    assert "- memory preference/4 用户喜欢先给结论再补充原因。" in text
    assert "- memory fact/5 用户正在搭建一个会持续学习的 agent。" in text


def test_format_checkpoint_diff_highlights_message_mismatches():
    class CheckpointMessage:
        def __init__(self, type_, content):
            self.type = type_
            self.content = content

    transcript_messages = [
        ThreadMessage(id=1, thread_id="t10", role="user", content="谢谢", created_at="2026-07-02T00:00:00+00:00"),
        ThreadMessage(id=2, thread_id="t10", role="assistant", content="不客气。", created_at="2026-07-02T00:00:01+00:00"),
    ]
    checkpoint_messages = [
        CheckpointMessage("human", "谢谢"),
        CheckpointMessage("ai", "我还在思考。"),
        CheckpointMessage("ai", "额外的一条状态消息"),
    ]

    text = format_checkpoint_diff("t10", transcript_messages, checkpoint_messages)

    assert "Checkpoint diff: t10" in text
    assert "- status mismatch" in text
    assert "mismatch #2" in text
    assert "transcript assistant 不客气。" in text
    assert "checkpoint ai 我还在思考。" in text
    assert "checkpoint-only #3 ai 额外的一条状态消息" in text


def test_format_checkpoint_diff_marks_match_state():
    class CheckpointMessage:
        def __init__(self, type_, content):
            self.type = type_
            self.content = content

    transcript_messages = [
        ThreadMessage(id=1, thread_id="t10", role="human", content="谢谢", created_at="2026-07-02T00:00:00+00:00"),
    ]
    checkpoint_messages = [CheckpointMessage("human", "谢谢")]

    text = format_checkpoint_diff("t10", transcript_messages, checkpoint_messages)

    assert text == "Checkpoint diff: t10\n- status match\nNo transcript/checkpoint differences."


def test_format_checkpoint_diff_marks_transcript_only_state():
    transcript_messages = [
        ThreadMessage(id=1, thread_id="t10", role="user", content="谢谢", created_at="2026-07-02T00:00:00+00:00"),
    ]

    text = format_checkpoint_diff("t10", transcript_messages, [])

    assert "Checkpoint diff: t10" in text
    assert "- status transcript-only" in text


def test_format_checkpoint_diff_marks_checkpoint_only_state():
    class CheckpointMessage:
        def __init__(self, type_, content):
            self.type = type_
            self.content = content

    text = format_checkpoint_diff("t10", [], [CheckpointMessage("ai", "我还在思考。")])

    assert "Checkpoint diff: t10" in text
    assert "- status checkpoint-only" in text


def test_format_checkpoint_diff_marks_empty_state():
    text = format_checkpoint_diff("t10", [], [])

    assert text == "Checkpoint diff: t10\n- status empty\nNo transcript/checkpoint differences."


def test_format_retrieval_comparison_shows_latest_audit_and_checkpoint_memories():
    retrieval_events = [
        RetrievalEvent(
            id=1,
            user_id="alice",
            thread_id="t10",
            user_text="你还记得我的回答偏好吗？",
            memory_count=1,
            memory_ids=[7],
            memory_preview="用户喜欢先给结论再补充原因。",
            created_at="2026-07-02T00:00:15+00:00",
        )
    ]

    text = format_retrieval_comparison(
        "t10",
        retrieval_events,
        [
            {
                "category": "preference",
                "content": "用户喜欢先给结论再补充原因。",
                "importance": 4,
                "source": "conversation",
            }
        ],
    )

    assert "Retrieval compare: t10" in text
    assert "- status match" in text
    assert "- audit memories=1 ids=7 preview=用户喜欢先给结论再补充原因。" in text
    assert "- checkpoint memories=1" in text
    assert "- checkpoint memory preference/4 用户喜欢先给结论再补充原因。" in text


def test_format_retrieval_comparison_flags_count_and_preview_mismatch():
    retrieval_events = [
        RetrievalEvent(
            id=1,
            user_id="alice",
            thread_id="t10",
            user_text="你还记得我的回答偏好吗？",
            memory_count=2,
            memory_ids=[7, 8],
            memory_preview="用户喜欢先给结论再补充原因。",
            created_at="2026-07-02T00:00:15+00:00",
        )
    ]

    text = format_retrieval_comparison(
        "t10",
        retrieval_events,
        [
            {
                "category": "fact",
                "content": "用户正在搭建一个会持续学习的 agent。",
                "importance": 5,
                "source": "conversation",
            }
        ],
    )

    assert "- status mismatch count, preview" in text


def test_format_retrieval_comparison_flags_audit_only_state():
    retrieval_events = [
        RetrievalEvent(
            id=1,
            user_id="alice",
            thread_id="t10",
            user_text="你还记得我的回答偏好吗？",
            memory_count=1,
            memory_ids=[7],
            memory_preview="用户喜欢先给结论再补充原因。",
            created_at="2026-07-02T00:00:15+00:00",
        )
    ]

    text = format_retrieval_comparison("t10", retrieval_events, [])

    assert "- status audit-only" in text


def test_format_retrieval_comparison_flags_checkpoint_only_state():
    text = format_retrieval_comparison(
        "t10",
        [],
        [
            {
                "category": "fact",
                "content": "用户正在搭建一个会持续学习的 agent。",
                "importance": 5,
                "source": "conversation",
            }
        ],
    )

    assert "- status checkpoint-only" in text


def test_format_retrieval_comparison_keeps_empty_state_message():
    text = format_retrieval_comparison("t10", [], [])

    assert text == "Retrieval compare: t10\nNo retrieval/checkpoint data."


def test_format_thread_inspection_renders_from_unified_inspection():
    class CheckpointMessage:
        def __init__(self, type_, content):
            self.type = type_
            self.content = content

    inspection = ThreadInspection(
        thread_id="t10",
        transcript_messages=[
            ThreadMessage(id=1, thread_id="t10", role="user", content="谢谢", created_at="2026-07-02T00:00:00+00:00"),
        ],
        routing_events=[
            RoutingEvent(
                id=1,
                user_id="alice",
                thread_id="t10",
                user_text="谢谢",
                should_retrieve=False,
                retrieve_reason="low_signal",
                should_learn=False,
                learn_reason="low_signal",
                created_at="2026-07-02T00:00:00+00:00",
            )
        ],
        retrieval_events=[],
        learning_events=[],
        dedupe_events=[
            DedupeEvent(
                id=1,
                user_id="alice",
                thread_id="t10",
                removed_count=1,
                removed_ids=[7],
                kept_ids=[3],
                created_at="2026-07-02T00:00:45+00:00",
            )
        ],
        memory_evolution_events=[
            MemoryEvolutionEvent(
                id=1,
                user_id="alice",
                thread_id="t10",
                action="revise",
                candidate_category="preference",
                candidate_content="以后回答先给结论。",
                target_memory_id=3,
                result_memory_id=4,
                reason="correction_phrase",
                created_at="2026-07-02T00:00:40+00:00",
            )
        ],
        checkpoint_available=True,
        checkpoint_retrieved_memories=[
            {
                "category": "preference",
                "content": "用户喜欢先给结论再补充原因。",
                "importance": 4,
                "source": "conversation",
            }
        ],
        checkpoint_messages=[CheckpointMessage("human", "谢谢")],
    )

    text = format_thread_inspection(inspection)

    assert "Thread audit: t10" in text
    assert "dedupe removed=1 ids=7 kept=3" in text
    assert "memory_evolution revise category=preference target=3 result=4 reason=correction_phrase" in text
    assert "candidate=以后回答先给结论。" in text
    assert "Checkpoint state: t10" in text
    assert "Retrieval compare: t10" in text
    assert "Checkpoint diff: t10" in text


def test_format_thread_inspection_hides_checkpoint_diff_when_runtime_lacks_checkpoint():
    inspection = ThreadInspection(
        thread_id="t10",
        transcript_messages=[
            ThreadMessage(id=1, thread_id="t10", role="user", content="谢谢", created_at="2026-07-02T00:00:00+00:00"),
        ],
        checkpoint_available=False,
        checkpoint_messages=[],
    )

    text = format_thread_inspection(inspection)

    assert "Checkpoint state: t10" in text
    assert "Checkpoint not available for this runtime." in text
    assert "Checkpoint diff: t10" not in text


def test_format_thread_inspection_omits_dedupe_events_from_other_threads():
    inspection = ThreadInspection(
        thread_id="t10",
        transcript_messages=[
            ThreadMessage(id=1, thread_id="t10", role="user", content="谢谢", created_at="2026-07-02T00:00:00+00:00"),
        ],
        dedupe_events=[
            DedupeEvent(
                id=1,
                user_id="alice",
                thread_id="t11",
                removed_count=1,
                removed_ids=[7],
                kept_ids=[3],
                created_at="2026-07-02T00:00:45+00:00",
            )
        ],
    )

    text = format_thread_inspection(inspection)

    assert "dedupe removed=" not in text


def test_build_agent_wraps_memory_store_for_langgraph_backend(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )

    agent, cli_store = build_agent(config)

    assert isinstance(agent, LangGraphAgent)
    assert isinstance(cli_store, SqliteCliStore)
    assert isinstance(cli_store.long_term_store, SqliteLongTermStore)
    assert isinstance(cli_store.long_term_store.semantic_memory_store, SqliteSemanticMemoryStore)
    assert isinstance(cli_store.long_term_store.profile_store, SqliteProfileStore)
    assert isinstance(cli_store.long_term_store.audit_store, SqliteAuditStore)
    assert isinstance(cli_store.long_term_store.transcript_store, SqliteTranscriptStore)


def test_build_runtime_returns_explicit_classic_dependencies(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="classic",
    )

    runtime = build_runtime(config)

    assert isinstance(runtime.agent, ConversationRuntime)
    assert isinstance(runtime.agent, ThreadInspectionRuntime)
    assert runtime.cli_store is runtime.memory_store
    assert runtime.long_term_store is None
    inspection = runtime.agent.inspect_thread("t1", user_id="alice")
    assert isinstance(inspection, ThreadInspection)
    assert inspection.thread_id == "t1"
    assert inspection.transcript_messages == []
    assert inspection.routing_events == []
    assert inspection.retrieval_events == []
    assert inspection.learning_events == []
    assert inspection.dedupe_events == []
    assert inspection.checkpoint_available is False
    assert inspection.checkpoint_messages == []
    runtime.agent.close()


def test_build_runtime_returns_explicit_langgraph_dependencies(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )

    runtime = build_runtime(config)

    assert isinstance(runtime.agent, LangGraphAgent)
    assert isinstance(runtime.agent, ThreadInspectionRuntime)
    assert isinstance(runtime.cli_store, SqliteCliStore)
    assert isinstance(runtime.long_term_store, SqliteLongTermStore)
    assert runtime.cli_store.long_term_store is runtime.long_term_store
    assert runtime.agent.semantic_memory_store is runtime.long_term_store.semantic_memory_store
    assert runtime.agent.profile_store is runtime.long_term_store.profile_store
    assert runtime.agent.audit_store is runtime.long_term_store.audit_store
    assert runtime.agent.transcript_store is runtime.long_term_store.transcript_store
    assert isinstance(runtime.agent.checkpoint_state_reader, LangGraphCheckpointStateReader)
    assert runtime.checkpoint_state_reader is runtime.agent.checkpoint_state_reader
    assert isinstance(runtime.thread_state_store, LangGraphThreadStateStore)
    assert runtime.agent.thread_state_store is runtime.thread_state_store
    assert runtime.thread_state_store.checkpoint_state_reader is runtime.checkpoint_state_reader
    assert runtime.thread_state_store.transcript_store is runtime.long_term_store.transcript_store


def test_build_runtime_does_not_configure_vector_indexer_without_zilliz_config(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )

    runtime = build_runtime(config)

    assert runtime.long_term_store.semantic_memory_store.vector_indexer is None


def test_build_runtime_configures_classic_vector_indexer_and_searcher_with_zilliz_config(tmp_path, monkeypatch):
    created_indexers = []
    created_searchers = []

    class TrackingVectorMemoryIndexer:
        def __init__(self, config):
            created_indexers.append(config)

    class TrackingVectorMemorySearcher:
        def __init__(self, config):
            created_searchers.append(config)

    monkeypatch.setattr("agent_app.bootstrap.VectorMemoryIndexer", TrackingVectorMemoryIndexer)
    monkeypatch.setattr("agent_app.bootstrap.VectorMemorySearcher", TrackingVectorMemorySearcher)

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="classic",
        zilliz_uri="https://example.zilliz.com.cn",
        zilliz_token="test-token",
    )

    runtime = build_runtime(config)

    assert created_indexers == [config]
    assert created_searchers == [config]
    assert isinstance(runtime.memory_store.semantic_store.vector_indexer, TrackingVectorMemoryIndexer)
    assert isinstance(runtime.memory_store.semantic_store.vector_searcher, TrackingVectorMemorySearcher)


def test_build_runtime_configures_vector_indexer_and_searcher_with_zilliz_config(tmp_path, monkeypatch):
    created_indexers = []
    created_searchers = []

    class TrackingVectorMemoryIndexer:
        def __init__(self, config):
            created_indexers.append(config)

    class TrackingVectorMemorySearcher:
        def __init__(self, config):
            created_searchers.append(config)

    monkeypatch.setattr("agent_app.bootstrap.VectorMemoryIndexer", TrackingVectorMemoryIndexer)
    monkeypatch.setattr("agent_app.bootstrap.VectorMemorySearcher", TrackingVectorMemorySearcher)

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
        zilliz_uri="https://example.zilliz.com.cn",
        zilliz_token="test-token",
    )

    runtime = build_runtime(config)

    assert created_indexers == [config]
    assert created_searchers == [config]
    assert isinstance(runtime.long_term_store.semantic_memory_store.vector_indexer, TrackingVectorMemoryIndexer)
    assert isinstance(runtime.long_term_store.semantic_memory_store.vector_searcher, TrackingVectorMemorySearcher)


def test_build_runtime_uses_langgraph_thread_inspection_builder(tmp_path, monkeypatch):
    created = []

    class TrackingBuilder:
        def __init__(self, cli_store, thread_state_store):
            created.append((cli_store, thread_state_store))

        def build(self, thread_id: str, user_id: str = "default"):
            return ThreadInspection(thread_id=thread_id, checkpoint_available=True)

    monkeypatch.setattr("agent_app.bootstrap.LangGraphThreadInspectionBuilder", TrackingBuilder)

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )

    runtime = build_runtime(config)

    assert isinstance(runtime.agent, LangGraphAgent)
    assert created == [(runtime.cli_store, runtime.thread_state_store)]


def test_build_runtime_langgraph_thread_state_store_exposes_snapshot_reader(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )

    runtime = build_runtime(config)
    snapshot = runtime.thread_state_store.get_thread_snapshot("t-inspect", user_id="alice")

    assert snapshot.state_keys == []
    assert snapshot.message_count == 0
    assert snapshot.updated_at is None


def test_thread_inspection_runtime_returns_checkpoint_messages(tmp_path):
    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )

    runtime = build_runtime(config)
    inspection = runtime.agent.inspect_thread("t-inspect", user_id="alice")

    assert isinstance(inspection, ThreadInspection)
    assert inspection.thread_id == "t-inspect"
    assert inspection.transcript_messages == []
    assert inspection.routing_events == []
    assert inspection.retrieval_events == []
    assert inspection.learning_events == []
    assert inspection.dedupe_events == []
    assert inspection.checkpoint_available is True
    assert inspection.checkpoint_state_keys == []
    assert inspection.checkpoint_message_count == 0
    assert inspection.checkpoint_step is None
    assert inspection.checkpoint_updated_at is None
    assert inspection.checkpoint_messages == []


def test_build_runtime_thread_inspection_includes_checkpoint_snapshot_metadata_after_reply(tmp_path, monkeypatch):
    class FakeGraphModel:
        def invoke(self, messages):
            return "好的。"

    monkeypatch.setattr("agent_app.bootstrap.LangChainQwenClient", lambda config: FakeGraphModel())

    config = AgentConfig(
        api_key="test-key",
        base_url="https://example.test/compatible-mode/v1",
        memory_db_path=tmp_path / "agent.db",
        checkpoint_db_path=tmp_path / "checkpoints.db",
        backend="langgraph",
    )

    runtime = build_runtime(config)
    runtime.agent.reply("谢谢", thread_id="t-inspect", user_id="alice")
    inspection = runtime.agent.inspect_thread("t-inspect", user_id="alice")

    assert inspection.checkpoint_available is True
    assert inspection.checkpoint_state_keys == ["messages", "retrieved_memories", "routing_decision"]
    assert inspection.checkpoint_message_count == 2
    assert inspection.checkpoint_step == 1
    assert inspection.checkpoint_updated_at is not None
    assert inspection.checkpoint_routing_decision == {
        "should_retrieve": False,
        "retrieve_reason": "low_signal",
        "should_learn": False,
        "learn_reason": "low_signal",
    }
    assert inspection.checkpoint_retrieved_memories == []
