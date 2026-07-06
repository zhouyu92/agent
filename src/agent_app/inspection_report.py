from __future__ import annotations

from .memory import DedupeEvent, LearningEvent, MemoryEvolutionEvent, RetrievalEvent, RoutingEvent, ThreadMessage
from .runtime_agent import ThreadInspection


def format_audit_timeline(
    thread_id: str,
    messages: list[ThreadMessage],
    routing_events: list[RoutingEvent],
    retrieval_events: list[RetrievalEvent],
    learning_events: list[LearningEvent],
    dedupe_events: list[DedupeEvent],
    memory_evolution_events: list[MemoryEvolutionEvent] | None = None,
) -> str:
    memory_evolution_events = memory_evolution_events or []
    timeline: list[tuple[str, str, str]] = []
    for message in messages:
        preview = message.content.replace("\n", " ")[:60]
        timeline.append(
            (
                message.created_at,
                "message",
                f"message {message.role} {preview}",
            )
        )
    for event in routing_events:
        preview = event.user_text.replace("\n", " ")[:60]
        timeline.append(
            (
                event.created_at,
                "routing",
                f"routing retrieve={event.should_retrieve}({event.retrieve_reason}) "
                f"learn={event.should_learn}({event.learn_reason}) user={preview}",
            )
        )
    for event in retrieval_events:
        preview = event.memory_preview.replace("\n", " ")[:80] or "none"
        memory_ids = ",".join(str(memory_id) for memory_id in event.memory_ids) or "none"
        timeline.append(
            (
                event.created_at,
                "retrieval",
                f"retrieval memories={event.memory_count} ids={memory_ids} preview={preview}",
            )
        )
    for event in learning_events:
        fields = ",".join(event.profile_fields) if event.profile_fields else "none"
        preview = event.user_text.replace("\n", " ")[:60]
        timeline.append(
            (
                event.created_at,
                "learning",
                f"learning memories={event.memory_count} profile={fields} user={preview}",
            )
        )
    for event in dedupe_events:
        if event.thread_id != thread_id:
            continue
        removed_ids = ",".join(str(memory_id) for memory_id in event.removed_ids) or "none"
        kept_ids = ",".join(str(memory_id) for memory_id in event.kept_ids) or "none"
        timeline.append(
            (
                event.created_at,
                "dedupe",
                f"dedupe removed={event.removed_count} ids={removed_ids} kept={kept_ids}",
            )
        )
    for event in memory_evolution_events:
        if event.thread_id != thread_id:
            continue
        target_memory_id = event.target_memory_id if event.target_memory_id is not None else "none"
        result_memory_id = event.result_memory_id if event.result_memory_id is not None else "none"
        timeline.append(
            (
                event.created_at,
                "memory_evolution",
                f"memory_evolution {event.action} target={target_memory_id} result={result_memory_id} reason={event.reason}",
            )
        )

    timeline.sort(key=lambda item: item[0])
    if not timeline:
        return f"Thread audit: {thread_id}\nNo audit events."
    lines = [f"Thread audit: {thread_id}"]
    lines.extend(f"- [{created_at}] {content}" for created_at, _, content in timeline)
    return "\n".join(lines)


def format_checkpoint_messages(
    thread_id: str,
    messages: list[object],
    checkpoint_available: bool = True,
    checkpoint_state_keys: list[str] | None = None,
    checkpoint_message_count: int | None = None,
    checkpoint_step: int | None = None,
    checkpoint_updated_at: str | None = None,
    checkpoint_routing_decision: dict[str, object] | None = None,
    checkpoint_retrieved_memories: list[dict[str, object]] | None = None,
) -> str:
    if not checkpoint_available:
        return f"Checkpoint state: {thread_id}\nCheckpoint not available for this runtime."

    lines = [f"Checkpoint state: {thread_id}"]
    if checkpoint_state_keys:
        lines.append(f"- keys {', '.join(checkpoint_state_keys)}")
    if checkpoint_message_count is not None:
        lines.append(f"- messages {checkpoint_message_count}")
    if checkpoint_step is not None:
        lines.append(f"- step {checkpoint_step}")
    if checkpoint_updated_at:
        lines.append(f"- updated_at {checkpoint_updated_at}")
    if checkpoint_routing_decision is not None:
        lines.append(
            "- routing "
            f"retrieve={checkpoint_routing_decision.get('should_retrieve')}({checkpoint_routing_decision.get('retrieve_reason')}) "
            f"learn={checkpoint_routing_decision.get('should_learn')}({checkpoint_routing_decision.get('learn_reason')})"
        )
    if checkpoint_retrieved_memories is not None:
        lines.append(f"- retrieved_memories {len(checkpoint_retrieved_memories)}")
        for memory in checkpoint_retrieved_memories[:3]:
            category = str(memory.get("category", "unknown"))
            importance = str(memory.get("importance", ""))
            content = str(memory.get("content", "")).replace("\n", " ")[:80]
            lines.append(f"- memory {category}/{importance} {content}")
    if not messages:
        lines.append("No checkpoint messages.")
        return "\n".join(lines)

    for message in messages:
        role = getattr(message, "type", message.__class__.__name__.lower())
        content = str(getattr(message, "content", "")).replace("\n", " ")[:80]
        lines.append(f"- {role} {content}")
    return "\n".join(lines)


def format_checkpoint_diff(thread_id: str, transcript_messages: list[ThreadMessage], checkpoint_messages: list[object]) -> str:
    transcript_pairs = [(message.role, message.content.replace("\n", " ")[:80]) for message in transcript_messages]
    checkpoint_pairs = [
        (
            getattr(message, "type", message.__class__.__name__.lower()),
            str(getattr(message, "content", "")).replace("\n", " ")[:80],
        )
        for message in checkpoint_messages
    ]

    if not transcript_pairs and not checkpoint_pairs:
        return f"Checkpoint diff: {thread_id}\n- status empty\nNo transcript/checkpoint differences."

    if transcript_pairs == checkpoint_pairs:
        return f"Checkpoint diff: {thread_id}\n- status match\nNo transcript/checkpoint differences."

    lines = [f"Checkpoint diff: {thread_id}"]
    if transcript_pairs and checkpoint_pairs:
        lines.append("- status mismatch")
    elif transcript_pairs:
        lines.append("- status transcript-only")
    elif checkpoint_pairs:
        lines.append("- status checkpoint-only")
    max_len = max(len(transcript_pairs), len(checkpoint_pairs))
    for index in range(max_len):
        transcript_pair = transcript_pairs[index] if index < len(transcript_pairs) else None
        checkpoint_pair = checkpoint_pairs[index] if index < len(checkpoint_pairs) else None
        display_index = index + 1
        if transcript_pair == checkpoint_pair:
            continue
        if transcript_pair is None and checkpoint_pair is not None:
            lines.append(f"- checkpoint-only #{display_index} {checkpoint_pair[0]} {checkpoint_pair[1]}")
            continue
        if checkpoint_pair is None and transcript_pair is not None:
            lines.append(f"- transcript-only #{display_index} {transcript_pair[0]} {transcript_pair[1]}")
            continue
        lines.append(
            f"- mismatch #{display_index} transcript {transcript_pair[0]} {transcript_pair[1]} | "
            f"checkpoint {checkpoint_pair[0]} {checkpoint_pair[1]}"
        )
    return "\n".join(lines)


def format_retrieval_comparison(
    thread_id: str,
    retrieval_events: list[RetrievalEvent],
    checkpoint_retrieved_memories: list[dict[str, object]],
) -> str:
    if not retrieval_events and not checkpoint_retrieved_memories:
        return f"Retrieval compare: {thread_id}\nNo retrieval/checkpoint data."

    lines = [f"Retrieval compare: {thread_id}"]
    if retrieval_events:
        latest = retrieval_events[-1]
        memory_ids = ",".join(str(memory_id) for memory_id in latest.memory_ids) or "none"
        preview = latest.memory_preview.replace("\n", " ")[:80] or "none"
        if not checkpoint_retrieved_memories:
            lines.append("- status audit-only")
        else:
            mismatch_reasons: list[str] = []
            if latest.memory_count != len(checkpoint_retrieved_memories):
                mismatch_reasons.append("count")
            checkpoint_preview = " | ".join(
                str(memory.get("content", "")).replace("\n", " ")[:80] for memory in checkpoint_retrieved_memories[:3]
            ) or "none"
            if preview != checkpoint_preview:
                mismatch_reasons.append("preview")
            if mismatch_reasons:
                lines.append(f"- status mismatch {', '.join(mismatch_reasons)}")
            else:
                lines.append("- status match")
        lines.append(f"- audit memories={latest.memory_count} ids={memory_ids} preview={preview}")
    else:
        if checkpoint_retrieved_memories:
            lines.append("- status checkpoint-only")
        lines.append("- audit none")

    lines.append(f"- checkpoint memories={len(checkpoint_retrieved_memories)}")
    for memory in checkpoint_retrieved_memories[:3]:
        category = str(memory.get("category", "unknown"))
        importance = str(memory.get("importance", ""))
        content = str(memory.get("content", "")).replace("\n", " ")[:80]
        lines.append(f"- checkpoint memory {category}/{importance} {content}")
    return "\n".join(lines)


def format_thread_inspection(inspection: ThreadInspection) -> str:
    sections = [
        format_audit_timeline(
            inspection.thread_id,
            inspection.transcript_messages,
            inspection.routing_events,
            inspection.retrieval_events,
            inspection.learning_events,
            inspection.dedupe_events,
            inspection.memory_evolution_events,
        ),
        format_checkpoint_messages(
            inspection.thread_id,
            inspection.checkpoint_messages,
            checkpoint_available=inspection.checkpoint_available,
            checkpoint_state_keys=inspection.checkpoint_state_keys,
            checkpoint_message_count=inspection.checkpoint_message_count,
            checkpoint_step=inspection.checkpoint_step,
            checkpoint_updated_at=inspection.checkpoint_updated_at,
            checkpoint_routing_decision=inspection.checkpoint_routing_decision,
            checkpoint_retrieved_memories=inspection.checkpoint_retrieved_memories,
        ),
    ]
    if inspection.checkpoint_available:
        sections.append(
            format_retrieval_comparison(
                inspection.thread_id,
                inspection.retrieval_events,
                inspection.checkpoint_retrieved_memories,
            )
        )
        sections.append(
            format_checkpoint_diff(inspection.thread_id, inspection.transcript_messages, inspection.checkpoint_messages)
        )
    return "\n\n".join(section for section in sections if section)
