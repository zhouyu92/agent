# Memory Evolution MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a production-leaning long-term memory evolution path so the agent can add, reinforce, revise, and ignore learned memories while keeping history and defaulting retrieval to active memories only.

**Architecture:** Extend the existing SQLite-backed memory model with active/superseded state and a dedicated memory evolution audit log. Add one semantic-store evolution entrypoint shared by both classic and LangGraph learning flows, keep decision logic in a focused helper module, and surface evolution events through existing thread inspection rather than adding a broad new CLI area.

**Tech Stack:** Python 3.10, SQLite, pytest, existing `agent_app` store/repository split, existing classic and LangGraph runtimes.

---

## File Structure

### Create

- `D:/agent/src/agent_app/memory_evolution.py`  
  Encapsulate evolution action types, structured results, and rule-based decision helpers.

- `D:/agent/docs/superpowers/plans/2026-07-06-memory-evolution-mvp.md`  
  This implementation plan.

### Modify

- `D:/agent/src/agent_app/memory.py`  
  Add dataclasses for memory evolution results and audit events; extend memory item state fields as needed.

- `D:/agent/src/agent_app/sqlite_schema.py`  
  Add `memories` evolution columns and create the `memory_evolution_events` table.

- `D:/agent/src/agent_app/semantic_memory_repository.py`  
  Add repository primitives for active-only lookups, status transitions, reinforcement updates, and inserting revision-linked memories.

- `D:/agent/src/agent_app/semantic_store.py`  
  Keep `add_memory()` for low-level insertions; add the shared `evolve_memory()` flow and make default retrieval only use active memories.

- `D:/agent/src/agent_app/sqlite_records.py`  
  Add repository support for memory evolution audit writes and reads.

- `D:/agent/src/agent_app/record_stores.py`  
  Expose memory evolution audit store methods.

- `D:/agent/src/agent_app/store.py`  
  Extend protocols and long-term store/CLI wrappers with evolution event access and shared evolution API.

- `D:/agent/src/agent_app/agent.py`  
  Route classic learning through the new evolution API instead of raw `add_memory()`.

- `D:/agent/src/agent_app/langgraph_agent.py`  
  Route LangGraph learning through the same evolution API and audit wiring.

- `D:/agent/src/agent_app/runtime_agent.py`  
  Extend thread inspection payload with memory evolution events.

- `D:/agent/src/agent_app/thread_inspection.py`  
  Include memory evolution events in unified inspection building.

- `D:/agent/src/agent_app/inspection_report.py`  
  Render memory evolution audit entries in `/thread` output.

- `D:/agent/src/agent_app/cli.py`  
  Keep command surface minimal; only consume the new thread inspection output and any new formatting helpers if needed.

- `D:/agent/README.md`  
  Document the new memory evolution behavior and the fact that thread inspection includes evolution audit entries.

### Tests

- `D:/agent/tests/test_memory.py`
- `D:/agent/tests/test_store.py`
- `D:/agent/tests/test_agent.py`
- `D:/agent/tests/test_langgraph_agent.py`
- `D:/agent/tests/test_cli.py`

## Task 1: Add evolution data model and schema

**Files:**
- Create: none
- Modify: `D:/agent/src/agent_app/memory.py`, `D:/agent/src/agent_app/sqlite_schema.py`
- Test: `D:/agent/tests/test_memory.py`, `D:/agent/tests/test_store.py`

- [ ] **Step 1: Write the failing tests for new dataclasses and schema-backed defaults**

Add or extend tests so they assert:

```python
def test_recent_memories_exclude_superseded_by_default(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")
    repository = store.semantic_store.repository
    first_id = repository.insert_memory(
        user_id="alice",
        category="preference",
        content="用户喜欢先给结论。",
        importance=4,
        source="conversation",
        created_at="2026-07-06T00:00:00+00:00",
    )
    repository.mark_memory_superseded(first_id)
    assert store.recent_memories(user_id="alice", limit=10) == []


def test_memory_evolution_event_round_trip(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")
    store.add_memory_evolution_event(
        user_id="alice",
        thread_id="t1",
        action="revise",
        candidate_category="preference",
        candidate_content="以后回答先给结论再补原因。",
        target_memory_id=1,
        result_memory_id=2,
        reason="correction_phrase",
    )
    events = store.recent_memory_evolution_events(user_id="alice", limit=1, thread_id="t1")
    assert events[0].action == "revise"
    assert events[0].target_memory_id == 1
    assert events[0].result_memory_id == 2
```

- [ ] **Step 2: Run the targeted tests to verify failure**

Run:

```bash
python -m pytest tests/test_memory.py tests/test_store.py -k evolution -q
```

Expected: FAIL with missing dataclasses, missing schema fields, or missing store methods.

- [ ] **Step 3: Add the minimal dataclasses and schema changes**

Implement the new structures in `memory.py` and `sqlite_schema.py`:

```python
@dataclass(frozen=True)
class MemoryEvolutionResult:
    action: str
    candidate_category: str
    candidate_content: str
    target_memory_id: int | None
    result_memory_id: int | None
    reason: str


@dataclass(frozen=True)
class MemoryEvolutionEvent:
    id: int
    user_id: str
    thread_id: str | None
    action: str
    candidate_category: str
    candidate_content: str
    target_memory_id: int | None
    result_memory_id: int | None
    reason: str
    created_at: str
```

```python
conn.execute(
    """
    CREATE TABLE IF NOT EXISTS memory_evolution_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT NOT NULL,
        thread_id TEXT,
        action TEXT NOT NULL,
        candidate_category TEXT NOT NULL,
        candidate_content TEXT NOT NULL,
        target_memory_id INTEGER,
        result_memory_id INTEGER,
        reason TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """
)
_ensure_column(conn, "memories", "status", "TEXT NOT NULL DEFAULT 'active'")
_ensure_column(conn, "memories", "supersedes_memory_id", "INTEGER")
_ensure_column(conn, "memories", "reinforcement_count", "INTEGER NOT NULL DEFAULT 0")
_ensure_column(conn, "memories", "last_reinforced_at", "TEXT")
```

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run:

```bash
python -m pytest tests/test_memory.py tests/test_store.py -k evolution -q
```

Expected: PASS for the new schema/dataclass coverage.

- [ ] **Step 5: Commit**

```bash
git add src/agent_app/memory.py src/agent_app/sqlite_schema.py tests/test_memory.py tests/test_store.py docs/superpowers/plans/2026-07-06-memory-evolution-mvp.md
git commit -m "feat: add memory evolution schema"
```

## Task 2: Add repository/store support for active memories, reinforce, and revise

**Files:**
- Create: `D:/agent/src/agent_app/memory_evolution.py`
- Modify: `D:/agent/src/agent_app/semantic_memory_repository.py`, `D:/agent/src/agent_app/semantic_store.py`, `D:/agent/src/agent_app/memory.py`
- Test: `D:/agent/tests/test_memory.py`

- [ ] **Step 1: Write the failing semantic-store tests**

Add tests that lock the four MVP actions:

```python
def test_evolve_memory_adds_new_memory_when_no_active_match(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")
    result = store.semantic_store.evolve_memory(
        category="fact",
        content="用户正在搭建一个持续学习 agent。",
        importance=4,
        source="conversation",
        user_id="alice",
    )
    assert result.action == "add"
    assert result.target_memory_id is None
    assert result.result_memory_id is not None


def test_evolve_memory_reinforces_existing_memory(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")
    repository = store.semantic_store.repository
    memory_id = repository.insert_memory(
        user_id="alice",
        category="preference",
        content="用户喜欢先给结论。",
        importance=3,
        source="conversation",
        created_at="2026-07-06T00:00:00+00:00",
    )
    result = store.semantic_store.evolve_memory(
        category="preference",
        content="用户还是喜欢先给结论。",
        importance=3,
        source="conversation",
        user_id="alice",
    )
    refreshed = store.recent_memories(user_id="alice", limit=10)[0]
    assert result.action == "reinforce"
    assert result.target_memory_id == memory_id
    assert refreshed.id == memory_id


def test_evolve_memory_revises_existing_memory(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")
    repository = store.semantic_store.repository
    old_id = repository.insert_memory(
        user_id="alice",
        category="preference",
        content="用户喜欢详细铺垫后再给结论。",
        importance=3,
        source="conversation",
        created_at="2026-07-06T00:00:00+00:00",
    )
    result = store.semantic_store.evolve_memory(
        category="preference",
        content="以后回答不是先铺垫，而是先给结论。",
        importance=4,
        source="conversation",
        user_id="alice",
    )
    all_memories = store.semantic_store.repository.list_memories("alice")
    assert result.action == "revise"
    assert result.target_memory_id == old_id
    assert any(row["id"] == old_id and row["status"] == "superseded" for row in all_memories)


def test_evolve_memory_ignores_low_value_candidate(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")
    result = store.semantic_store.evolve_memory(
        category="general",
        content="好的",
        importance=1,
        source="conversation",
        user_id="alice",
    )
    assert result.action == "ignore"
    assert store.recent_memories(user_id="alice", limit=10) == []
```

- [ ] **Step 2: Run the semantic-store tests to verify failure**

Run:

```bash
python -m pytest tests/test_memory.py -k "evolve_memory or superseded" -q
```

Expected: FAIL because `evolve_memory()` and related repository methods do not exist yet.

- [ ] **Step 3: Implement the decision helper and repository primitives**

Create `memory_evolution.py` with explicit helper functions:

```python
CORRECTION_MARKERS = ("不是", "而是", "改成", "更准确地说", "实际上", "以后以", "以后回答")


def looks_like_correction(text: str) -> bool:
    lowered = text.strip()
    return any(marker in lowered for marker in CORRECTION_MARKERS)


def choose_evolution_action(candidate_content: str, matched_content: str) -> tuple[str, str]:
    if looks_like_correction(candidate_content):
        return "revise", "correction_phrase"
    if candidate_content.strip() == matched_content.strip():
        return "ignore", "no_new_information"
    return "reinforce", "confirmed_existing_memory"
```

Add repository methods shaped like:

```python
def active_memories(self, user_id: str) -> list[sqlite3.Row]: ...
def mark_memory_superseded(self, memory_id: int, user_id: str | None = None) -> None: ...
def reinforce_memory(self, memory_id: int, *, created_at: str, increase_importance: bool = True) -> None: ...
def insert_revision_memory(..., supersedes_memory_id: int | None = None) -> int: ...
```

Implement `SqliteSemanticMemoryStore.evolve_memory()` so it:

```python
matched = self._best_active_match(user_id, category, content)
if low_value:
    return MemoryEvolutionResult(action="ignore", ...)
if matched is None:
    inserted = self.repository.insert_revision_memory(..., supersedes_memory_id=None)
    return MemoryEvolutionResult(action="add", result_memory_id=inserted, ...)
action, reason = choose_evolution_action(content, matched["content"])
if action == "ignore":
    return MemoryEvolutionResult(action="ignore", target_memory_id=matched["id"], ...)
if action == "reinforce":
    self.repository.reinforce_memory(matched["id"], created_at=now_iso())
    return MemoryEvolutionResult(action="reinforce", target_memory_id=matched["id"], result_memory_id=matched["id"], ...)
self.repository.mark_memory_superseded(matched["id"], user_id=user_id)
new_id = self.repository.insert_revision_memory(..., supersedes_memory_id=matched["id"])
return MemoryEvolutionResult(action="revise", target_memory_id=matched["id"], result_memory_id=new_id, ...)
```

- [ ] **Step 4: Run the semantic-store tests to verify they pass**

Run:

```bash
python -m pytest tests/test_memory.py -k "evolve_memory or superseded" -q
```

Expected: PASS with all four actions covered.

- [ ] **Step 5: Commit**

```bash
git add src/agent_app/memory_evolution.py src/agent_app/semantic_memory_repository.py src/agent_app/semantic_store.py src/agent_app/memory.py tests/test_memory.py
git commit -m "feat: add memory evolution decisions"
```

## Task 3: Add audit storage and thread inspection support for memory evolution

**Files:**
- Modify: `D:/agent/src/agent_app/sqlite_records.py`, `D:/agent/src/agent_app/record_stores.py`, `D:/agent/src/agent_app/store.py`, `D:/agent/src/agent_app/runtime_agent.py`, `D:/agent/src/agent_app/thread_inspection.py`, `D:/agent/src/agent_app/inspection_report.py`
- Test: `D:/agent/tests/test_store.py`, `D:/agent/tests/test_cli.py`

- [ ] **Step 1: Write the failing audit and `/thread` tests**

Add tests shaped like:

```python
def test_memory_evolution_events_round_trip_from_sqlite_audit_store(tmp_path):
    memory = MemoryStore(tmp_path / "agent.db")
    store = SqliteAuditStore(memory)
    store.add_memory_evolution_event(
        user_id="alice",
        thread_id="t1",
        action="reinforce",
        candidate_category="preference",
        candidate_content="用户还是喜欢先给结论。",
        target_memory_id=3,
        result_memory_id=3,
        reason="confirmed_existing_memory",
    )
    event = store.recent_memory_evolution_events(user_id="alice", limit=1, thread_id="t1")[0]
    assert event.action == "reinforce"
    assert event.result_memory_id == 3


def test_format_thread_inspection_shows_memory_evolution_events():
    inspection = ThreadInspection(
        thread_id="t1",
        memory_evolution_events=[
            MemoryEvolutionEvent(
                id=1,
                user_id="alice",
                thread_id="t1",
                action="revise",
                candidate_category="preference",
                candidate_content="以后回答先给结论。",
                target_memory_id=3,
                result_memory_id=4,
                reason="correction_phrase",
                created_at="2026-07-06T00:00:00+00:00",
            )
        ],
    )
    text = format_thread_inspection(inspection)
    assert "memory_evolution revise target=3 result=4 reason=correction_phrase" in text
```

- [ ] **Step 2: Run the audit/CLI tests to verify failure**

Run:

```bash
python -m pytest tests/test_store.py tests/test_cli.py -k "memory_evolution or thread_inspection" -q
```

Expected: FAIL with missing audit methods, missing thread inspection fields, or missing formatter output.

- [ ] **Step 3: Implement the audit plumbing**

Add store/repository methods shaped like:

```python
def add_memory_evolution_event(
    self,
    *,
    user_id: str,
    thread_id: str | None,
    action: str,
    candidate_category: str,
    candidate_content: str,
    target_memory_id: int | None,
    result_memory_id: int | None,
    reason: str,
) -> None: ...


def recent_memory_evolution_events(
    self,
    user_id: str = "default",
    limit: int = 10,
    *,
    thread_id: str | None = None,
) -> list[MemoryEvolutionEvent]: ...
```

Extend thread inspection and formatter output:

```python
@dataclass
class ThreadInspection:
    ...
    memory_evolution_events: list[MemoryEvolutionEvent] = field(default_factory=list)
```

```python
timeline.append(
    (
        event.created_at,
        "memory_evolution",
        f"memory_evolution {event.action} target={event.target_memory_id or 'none'} "
        f"result={event.result_memory_id or 'none'} reason={event.reason}",
    )
)
```

- [ ] **Step 4: Run the audit/CLI tests to verify they pass**

Run:

```bash
python -m pytest tests/test_store.py tests/test_cli.py -k "memory_evolution or thread_inspection" -q
```

Expected: PASS with evolution audit round-trips and `/thread` rendering covered.

- [ ] **Step 5: Commit**

```bash
git add src/agent_app/sqlite_records.py src/agent_app/record_stores.py src/agent_app/store.py src/agent_app/runtime_agent.py src/agent_app/thread_inspection.py src/agent_app/inspection_report.py tests/test_store.py tests/test_cli.py
git commit -m "feat: add memory evolution audit"
```

## Task 4: Route both learning paths through the shared evolution API

**Files:**
- Modify: `D:/agent/src/agent_app/agent.py`, `D:/agent/src/agent_app/langgraph_agent.py`, `D:/agent/src/agent_app/store.py`
- Test: `D:/agent/tests/test_agent.py`, `D:/agent/tests/test_langgraph_agent.py`, `D:/agent/tests/test_memory.py`

- [ ] **Step 1: Write the failing integration tests**

Add tests that assert classic and LangGraph learning both call the same evolution entrypoint:

```python
def test_conversational_agent_learn_from_turn_uses_evolve_memory(tmp_path):
    calls = []

    class TrackingMemoryStore(MemoryStore):
        def evolve_memory(self, *, category, content, importance, source, user_id, thread_id=None):
            calls.append((category, content, importance, source, user_id, thread_id))
            return MemoryEvolutionResult(
                action="add",
                candidate_category=category,
                candidate_content=content,
                target_memory_id=None,
                result_memory_id=1,
                reason="new_memory",
            )

    ...
    assert calls == [("preference", "用户喜欢先给结论。", 4, "conversation", "alice", "t1")]
```

```python
def test_langgraph_learn_node_uses_shared_evolve_memory_api(...):
    ...
    assert recorded == [
        ("preference", "用户喜欢先给结论。", 4, "conversation", "alice", "t1")
    ]
```

- [ ] **Step 2: Run the integration tests to verify failure**

Run:

```bash
python -m pytest tests/test_agent.py tests/test_langgraph_agent.py -k evolve_memory -q
```

Expected: FAIL because both learning paths still call `add_memory()` directly.

- [ ] **Step 3: Implement the shared learning path**

Update classic learning:

```python
for item in update.memories:
    evolution = self.memory.evolve_memory(
        category=item.category,
        content=item.content,
        importance=item.importance,
        source="conversation",
        user_id=user_id,
        thread_id=thread_id,
    )
    if evolution.action != "ignore":
        saved_memory_count += 1
```

Update LangGraph learning similarly:

```python
evolution = self.semantic_memory_store.evolve_memory(
    category=item.category,
    content=item.content,
    importance=item.importance,
    source="conversation",
    user_id=user_id,
    thread_id=thread_id,
)
if evolution.action != "ignore":
    saved_memory_count += 1
self.audit_store.add_memory_evolution_event(...)
```

If the audit write is already done inside the shared semantic-store evolution method, keep it there and do not duplicate it in the runtimes.

- [ ] **Step 4: Run the integration tests to verify they pass**

Run:

```bash
python -m pytest tests/test_agent.py tests/test_langgraph_agent.py -k evolve_memory -q
```

Expected: PASS, proving both orchestration paths share one evolution API.

- [ ] **Step 5: Commit**

```bash
git add src/agent_app/agent.py src/agent_app/langgraph_agent.py src/agent_app/store.py tests/test_agent.py tests/test_langgraph_agent.py
git commit -m "feat: share memory evolution across runtimes"
```

## Task 5: Preserve retrieval behavior and document the feature

**Files:**
- Modify: `D:/agent/src/agent_app/semantic_store.py`, `D:/agent/README.md`
- Test: `D:/agent/tests/test_memory.py`, `D:/agent/tests/test_cli.py`

- [ ] **Step 1: Write the failing retrieval and documentation-facing tests**

Add a retrieval test that proves superseded memories stay out of default search:

```python
def test_search_memories_excludes_superseded_items(tmp_path):
    store = MemoryStore(tmp_path / "agent.db")
    repository = store.semantic_store.repository
    old_id = repository.insert_memory(
        user_id="alice",
        category="preference",
        content="用户喜欢先铺垫。",
        importance=3,
        source="conversation",
        created_at="2026-07-06T00:00:00+00:00",
    )
    repository.mark_memory_superseded(old_id, user_id="alice")
    repository.insert_revision_memory(
        user_id="alice",
        category="preference",
        content="用户喜欢先给结论。",
        importance=4,
        source="conversation",
        created_at="2026-07-06T00:01:00+00:00",
        supersedes_memory_id=old_id,
    )
    results = store.search_memories("结论", user_id="alice", limit=5)
    assert [item.content for item in results] == ["用户喜欢先给结论。"]
```

- [ ] **Step 2: Run the retrieval test to verify failure**

Run:

```bash
python -m pytest tests/test_memory.py -k "superseded and search_memories" -q
```

Expected: FAIL if search still includes non-active memories.

- [ ] **Step 3: Tighten retrieval and update docs**

Constrain semantic store lookups to active rows only:

```python
rows = self.repository.active_memories(user_id)
```

Document the behavior in `README.md` with text shaped like:

```md
- 长期记忆演化：学习阶段不会只做追加；系统会对候选记忆执行 add / reinforce / revise / ignore。
- 默认检索只使用 active 记忆；被修正的 superseded 记忆保留用于审计，不进入默认回复上下文。
- `/thread <thread_id>` 现在也会显示 memory evolution 审计事件。
```

- [ ] **Step 4: Run the retrieval/documentation-adjacent tests to verify they pass**

Run:

```bash
python -m pytest tests/test_memory.py tests/test_cli.py -k "superseded or thread_inspection" -q
```

Expected: PASS with retrieval and inspection output intact.

- [ ] **Step 5: Commit**

```bash
git add src/agent_app/semantic_store.py README.md tests/test_memory.py tests/test_cli.py
git commit -m "docs: document memory evolution behavior"
```

## Task 6: Full regression verification

**Files:**
- Modify: none
- Test: `D:/agent/tests/test_memory.py`, `D:/agent/tests/test_store.py`, `D:/agent/tests/test_agent.py`, `D:/agent/tests/test_langgraph_agent.py`, `D:/agent/tests/test_cli.py`

- [ ] **Step 1: Run focused evolution regression**

Run:

```bash
python -m pytest tests/test_memory.py tests/test_store.py tests/test_agent.py tests/test_langgraph_agent.py tests/test_cli.py -k "evolution or superseded or thread_inspection" -q
```

Expected: PASS.

- [ ] **Step 2: Run the broader project test suite**

Run:

```bash
python -m pytest -q
```

Expected: PASS for the full suite, including pre-existing memory governance and LangGraph tests.

- [ ] **Step 3: Run compile verification**

Run:

```bash
python -m compileall src tests
```

Expected: PASS with all touched modules compiling cleanly.

- [ ] **Step 4: Review the final user-facing scope**

Check these requirements against current behavior:

```text
1. add / reinforce / revise / ignore all covered by tests
2. active memories only in default retrieval
3. classic and langgraph both use evolve_memory
4. /thread exposes memory evolution audit
5. existing /memories, /forget, /dedupe-memories, /dedupe-log still pass
```

- [ ] **Step 5: Commit**

```bash
git add README.md src/agent_app tests
git commit -m "test: verify memory evolution mvp"
```

## Self-Review Notes

- Spec coverage:
  - `add / reinforce / revise / ignore`: Task 2 and Task 4
  - `active / superseded`: Task 1, Task 2, Task 5
  - shared classic/langgraph evolution path: Task 4
  - evolution audit and `/thread`: Task 3
  - regression safety and existing governance commands: Task 6
- Placeholder scan:
  - No `TODO`, `TBD`, or “similar to above” placeholders remain in task steps.
- Type consistency:
  - The plan uses `MemoryEvolutionResult`, `MemoryEvolutionEvent`, `evolve_memory()`, `add_memory_evolution_event()`, and `recent_memory_evolution_events()` consistently across tasks.
