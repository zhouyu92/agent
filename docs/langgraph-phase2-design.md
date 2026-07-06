# LangGraph Phase 2 Design

## Goal

在不推翻当前 MVP 的前提下，把现有 classic / langgraph 双路径逐步收敛到更清晰的生产级边界：

- `Checkpoint` 负责线程级短期状态
- `ThreadStateStore` 负责线程状态读写适配
- `LongTermStore` 负责长期记忆、画像、审计
- `bootstrap` 负责显式装配，不让依赖关系散落在入口代码里

## Current State

截至 2026-07-03，项目已经具备这些结构基础：

- `MemoryStore` 已经基本变成兼容 facade
- `SqliteSemanticMemoryStore`、`SqliteProfileStore`、`SqliteAuditStore`、`SqliteTranscriptStore` 已经独立
- `SqliteLongTermStore` 作为组合对象，持有上述 split stores
- `LangGraphAgent` 已支持显式注入：
  - `semantic_memory_store`
  - `profile_store`
  - `audit_store`
  - `transcript_store`
  - `thread_state_store`
- `build_runtime()` 已显式装配 `long_term_store` 和 `thread_state_store`

这意味着第二阶段不需要再先做一次大规模拆分，可以直接进入 LangGraph 语义上的边界对齐。

## Boundary Model

### 1. Checkpoint

`Checkpoint` 只负责 graph state：

- 当前线程消息状态
- routing 决策状态
- retrieval node 输出
- learn node 前后的图内状态

它不负责：

- 长期记忆语义检索
- agent profile
- learning / routing / retrieval 审计记录
- CLI 查询视图

### 2. Thread State

`ThreadStateStore` 是 `Checkpoint` 的应用层适配器，不是长期存储总线。

当前阶段它负责两件事：

- 从 graph state 读取线程消息
- 把最终 user / assistant turn 同步落到 transcript store

第二阶段建议把它稳定为一个明确契约：

- `record_turn(...)`
- `get_thread_messages(...)`
- 视需要新增 `get_checkpoint_state(...)`，但不要让 CLI 直接依赖 LangGraph 原始对象

### 3. Long-Term Store

`LongTermStore` 继续负责所有非 checkpoint 的可持久化长期信息：

- semantic memories
- profile
- audit events
- transcript 持久化

第二阶段里它仍然可以是 SQLite 实现，但不应承担 graph state 的职责。

## Migration Plan

### Step 1

保持当前 `LangGraphAgent` 图结构不变，只继续收紧装配边界。

完成标准：

- runtime 只从 `build_runtime()` 进入
- LangGraph 依赖都从 bootstrap 显式传入

这一步已经基本完成。

### Step 2

把 `thread_state_store` 和 LangGraph checkpoint 的关系明确化。

建议实现：

- 新增一个 checkpoint-facing adapter，而不是让 bootstrap 通过私有方法回填 reader
- 让 `LangGraphThreadStateStore` 接收一个稳定的 state reader 接口，而不是直接依赖 agent 私有实现细节

目标：

- bootstrap 不需要写入 `agent._get_graph_messages`
- thread state 读取路径不依赖私有方法约定

### Step 3

把 classic backend 的“短期上下文”也向统一 runtime 语义靠拢。

不是要强行把 classic 改成 LangGraph，而是让两者在装配层共享同一套概念：

- conversation runtime
- semantic store
- profile store
- audit store
- transcript store
- optional thread state store

目标：

- classic 和 langgraph 差异主要落在 orchestration 层
- 存储和查询边界保持一致

### Step 4

如果确认 LangGraph 成为主路径，再考虑把 classic backend 降级为兼容模式或测试模式。

这一步的前置条件：

- checkpoint/thread-state adapter 稳定
- CLI 查询口不再依赖 classic 特殊行为
- runtime bootstrap 已能清晰表达两条路径的差异

## Non-Goals

第二阶段不做这些事：

- 不引入新的生产数据库
- 不改模型供应商接口
- 不一次性替换整个 CLI
- 不把 audit、memory、checkpoint 混成一个“大一统 store”

## Acceptance Criteria

当第二阶段完成时，代码层面应该满足：

- `build_runtime()` 是唯一主装配入口
- LangGraph runtime 显式装配 checkpoint-related 和 long-term-related 依赖
- `ThreadStateStore` 不再通过 agent 私有方法回填核心读取能力
- CLI 查询路径不需要知道 backend 内部 graph 细节
- classic / langgraph 的差异主要体现在 orchestration，而不是存储边界

## Suggested Next Change

最自然的下一步实现是：

1. 为 checkpoint state reader 定义一个小接口
2. 让 `LangGraphThreadStateStore` 依赖这个接口
3. 去掉 bootstrap 里对 `agent._get_graph_messages` 的回填
