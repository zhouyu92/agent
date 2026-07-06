from agent_app.cli import (
    build_agent,
    filter_routing_events,
    format_audit_timeline,
    format_checkpoint_diff,
    format_retrieval_comparison,
    format_thread_inspection,
    format_memory_evolution_events,
    format_learning_events,
    main,
    parse_memory_evolution_log_query,
    parse_dedupe_log_query,
    parse_dedupe_query,
    parse_memory_query,
    format_memories,
    format_profile,
    format_routing_events,
    format_checkpoint_messages,
)
from agent_app.agent import ConversationalAgent
from agent_app.bootstrap import build_runtime
from agent_app.config import AgentConfig
from agent_app.langgraph_agent import LangGraphAgent
from agent_app.memory import AgentProfile, DedupeEvent, DedupeResult, LearningEvent, MemoryEvolutionEvent, MemoryItem, RetrievalEvent, RoutingEvent, ThreadMessage
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
    assert "/memories [category=<name>] [importance=<1-5>] [status=<active|superseded>] [query]" in output
    assert "/dedupe-memories [thread=<thread_id>]" in output
    assert "/dedupe-log [thread=<thread_id>] [limit=<n>]" in output
    assert "/memory-log [thread=<thread_id>] [action=<add|reinforce|revise|ignore>] [limit=<n>]" in output


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
        ):
            self.calls.append((user_id, limit, thread_id))
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
    assert fake_store.calls == [("alice", 10, "t10")]
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
        ):
            self.calls.append((user_id, limit, thread_id, action))
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
    assert fake_store.calls == [("alice", 10, None, "revise")]
    assert "action=revise" in output


def test_main_rejects_invalid_memory_log_filter(monkeypatch, capsys):
    class FakeAgent:
        def close(self):
            return None

    class FakeCliStore:
        def recent_memory_evolution_events(self, user_id: str, limit: int, thread_id: str | None = None):
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
    assert "Usage: /memory-log [thread=<thread_id>] [action=<add|reinforce|revise|ignore>] [limit=<n>]" in output


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


def test_main_filters_memories_by_category_and_importance(monkeypatch, capsys):
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
    assert "Usage: /memories [category=<name>] [importance=<1-5>] [status=<active|superseded>] [query]" in output


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
    assert "Usage: /memories [category=<name>] [importance=<1-5>] [status=<active|superseded>] [query]" in output


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
    assert "Usage: /memories [category=<name>] [importance=<1-5>] [status=<active|superseded>] [query]" in output


def test_parse_memory_query_extracts_filters_and_text_query():
    query, filters, error = parse_memory_query("category=preference importance=4 status=active 结论优先")

    assert query == "结论优先"
    assert filters == {
        "category": "preference",
        "importance": 4,
        "status": "active",
    }
    assert error is None


def test_parse_memory_query_rejects_unknown_status():
    query, filters, error = parse_memory_query("status=archived")

    assert query == ""
    assert filters == {}
    assert error == "Usage: /memories [category=<name>] [importance=<1-5>] [status=<active|superseded>] [query]"


def test_parse_memory_evolution_log_query_extracts_filters():
    filters, error = parse_memory_evolution_log_query("thread=t10 action=revise limit=5")

    assert filters == {"thread_id": "t10", "action": "revise", "limit": 5}
    assert error is None


def test_parse_memory_evolution_log_query_rejects_unknown_action():
    filters, error = parse_memory_evolution_log_query("action=archive")

    assert filters == {}
    assert error == "Usage: /memory-log [thread=<thread_id>] [action=<add|reinforce|revise|ignore>] [limit=<n>]"


def test_parse_memory_query_rejects_unknown_filter():
    query, filters, error = parse_memory_query("source=conversation")

    assert query == ""
    assert filters == {}
    assert error == "Usage: /memories [category=<name>] [importance=<1-5>] [status=<active|superseded>] [query]"


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
        )
    ]

    text = format_memories(memories)

    assert "#1" in text
    assert "[preference/4]" in text
    assert "[active]" in text
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
    assert "style_notes" in text


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

    filters = parse_routing_query("thread=t10 learn=false recall_turn")

    assert filters == {
        "thread_id": "t10",
        "learn": False,
        "retrieve": None,
        "reason": "recall_turn",
        "text_query": None,
    }


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
    assert "memory_evolution revise target=3 result=4 reason=correction_phrase" in text
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
