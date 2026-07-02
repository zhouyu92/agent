from __future__ import annotations

from .agent import ConversationalAgent
from .config import AgentConfig
from .llm import QwenClient
from .memory import AgentProfile, LearningEvent, MemoryItem, MemoryStore


def main() -> None:
    config = AgentConfig.from_env()
    memory = MemoryStore(config.memory_db_path)
    model = QwenClient(config)
    agent = ConversationalAgent(config, memory, model)
    user_id = config.user_id

    print(f"Agent ready for user '{user_id}'. Type /help for commands, /exit to quit.")
    while True:
        user_text = input("you> ").strip()
        if user_text in {"/exit", "/quit"}:
            break
        if user_text == "/help":
            print("Commands: /profile, /memories [query], /forget <memory_id>, /learning, /exit")
            continue
        if user_text == "/profile":
            print(format_profile(memory.get_profile()))
            continue
        if user_text.startswith("/memories"):
            query = user_text.removeprefix("/memories").strip()
            memories = (
                memory.search_memories(query, limit=10, user_id=user_id)
                if query
                else memory.recent_memories(limit=10, user_id=user_id)
            )
            print(format_memories(memories))
            continue
        if user_text == "/learning":
            print(format_learning_events(memory.recent_learning_events(user_id=user_id, limit=10)))
            continue
        if user_text.startswith("/forget"):
            raw_id = user_text.removeprefix("/forget").strip()
            if not raw_id.isdigit():
                print("Usage: /forget <memory_id>")
                continue
            deleted = memory.delete_memory(int(raw_id), user_id=user_id)
            print("Memory deleted." if deleted else "No matching memory for current user.")
            continue
        if not user_text:
            continue
        answer = agent.reply(user_text, user_id=user_id)
        print(f"agent> {answer}")


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
    return "\n".join(f"- #{item.id} [{item.category}/{item.importance}] {item.content}" for item in memories)


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
