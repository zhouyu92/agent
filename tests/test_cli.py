from agent_app.cli import format_learning_events, format_memories, format_profile
from agent_app.memory import AgentProfile, LearningEvent, MemoryItem


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
        )
    ]

    text = format_memories(memories)

    assert "#1" in text
    assert "[preference/4]" in text
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
