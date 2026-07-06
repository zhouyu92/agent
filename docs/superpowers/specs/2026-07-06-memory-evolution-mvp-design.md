# Memory Evolution MVP Design

## Goal

在现有本地 agent MVP 上，把长期记忆从“只会追加和人工去重”提升到“能够根据新对话对旧记忆做增量演化”，让 agent 更接近持续学习和自我修正，而不是单纯累积文本片段。

这个设计只覆盖长期记忆演化的 MVP，不扩展到更大的 LangGraph 编排重构，也不引入新的外部存储。

## Context

当前仓库已经具备这些基础：

- classic 和 langgraph 双后端
- SQLite 长期记忆、profile、audit、transcript 分层
- routing / retrieval / learning / dedupe 审计链路
- `/memories`、`/forget`、`/dedupe-memories`、`/dedupe-log`、`/thread` CLI 治理入口

当前长期记忆行为仍然偏 `append-only`：

- 学习步骤会从模型输出 `memories[]`
- 对每条候选记忆调用 `add_memory()`
- `add_memory()` 只做敏感信息过滤和“相似记忆拒绝”
- 如果已经写错或重复，需要依赖 `/forget` 或 `/dedupe-memories` 做人工治理

这意味着 agent 现在“会记住”，但还不会系统性地“修正自己以前的认知”。

## Problem

如果用户后续给出更准确、更新或更稳定的信息，agent 应该能区分以下几种情况：

1. 这是一个全新的长期记忆，应该新增
2. 这是对现有记忆的重复确认，应该强化原记忆而不是再写一条
3. 这是对旧记忆的修正，应该保留历史，但让新版本成为当前有效记忆
4. 这是低价值或噪声，不应该进入长期记忆

当前实现只覆盖了第 1 和第 4 的一部分，且第 1 会和“重复确认”混在一起，第 3 完全缺失。

## Non-Goals

这个 MVP 明确不做这些事情：

- 不引入向量数据库或新的生产依赖
- 不把 classic backend 全量迁移到 LangGraph
- 不做跨多条记忆的自动 summary / merge 压缩
- 不做复杂置信度学习或 reinforcement decay 算法
- 不把 CLI 改造成完整的记忆可视化工作台

## Approaches Considered

### Approach A: 在现有 memories 表上增加演化字段

思路：

- 继续以 `memories` 为主表
- 为每条记忆增加状态和版本关联字段
- 检索默认只读当前有效记忆
- 演化动作单独记录 audit

优点：

- 改动面最小
- 可以复用现有 search / CLI / audit / store 分层
- 对 classic 和 langgraph 后端都容易复用

缺点：

- 版本历史表达力弱于完整 revision ledger
- 后续如果要做复杂 lineage，可读性不如专门版本表

### Approach B: 单独建立 memory revisions / ledger

思路：

- 现有 memories 更像“逻辑记忆”
- 新增 revisions 表记录每次演化
- active revision 决定当前检索内容

优点：

- 审计和可追溯性最好
- 以后扩展 merge / rollback 更自然

缺点：

- 第一版复杂度明显偏高
- 会连带改动 repository、CLI、检索和测试结构

### Approach C: 只演化 profile，不演化 memory

思路：

- 暂时不动长期记忆结构
- 只让 profile 更容易被学习和修正

优点：

- 风险最低
- 可以更快增强“人格感”

缺点：

- 对“持续学习、自我修正”帮助有限
- 不能解决长期记忆会越积越乱的问题

## Recommendation

推荐采用 **Approach A**。

原因：

- 它最贴近当前仓库的成熟边界
- 可以在不推翻 SQLite store 结构的前提下，给长期记忆补上“强化 / 修正”的核心能力
- 是实现“像真人一样会更新认知”的最短路径

## Proposed Behavior

### Evolution actions

MVP 只支持四种动作：

- `add`: 新增一条长期记忆
- `reinforce`: 现有记忆被再次确认，提升其稳定性
- `revise`: 新信息修正旧记忆，旧记忆保留历史但不再是当前有效版本
- `ignore`: 候选内容噪声太高或价值太低，不写入

MVP 暂不支持 `merge`。这类动作往往要求模型把多条旧记忆压缩成一条新摘要，会显著增加歧义和测试复杂度。

### Conflict policy

默认冲突策略为：

- 保留旧记忆历史
- 新记忆如果被判定为修正版，则成为新的 `active` 版本
- 被修正的旧记忆标记为 `superseded`
- 检索默认只返回 `active` 记忆

这个策略既能保留可审计性，也能保证实际回复上下文尽量干净。

### Decision flow

当前学习流程：

`parse learning update -> add_memory`

目标流程：

`parse learning update -> retrieve candidate-related memories -> decide action -> apply action -> record evolution audit`

其中 “decide action” 的第一版不引入额外模型调用，而是优先复用现有的词项重叠和相似性规则，让 MVP 保持可解释和可测试。

## Data Model

### memories table changes

在现有 `memories` 表上增加这些字段：

- `status TEXT NOT NULL DEFAULT 'active'`
- `supersedes_memory_id INTEGER`
- `reinforcement_count INTEGER NOT NULL DEFAULT 0`
- `last_reinforced_at TEXT`

语义说明：

- `status='active'`: 当前可被检索和展示为默认结果
- `status='superseded'`: 历史保留，但默认不进入检索上下文
- `supersedes_memory_id`: 如果当前记忆是修正版，指向它替代的旧记忆
- `reinforcement_count`: 被重复确认的次数
- `last_reinforced_at`: 最近一次被强化的时间

MVP 不要求外键约束或复杂 lineage 查询，只要求这些字段可以被正确读写。

### New audit event

新增一类记忆演化审计事件，建议命名为 `memory_evolution_events`。

建议字段：

- `id`
- `user_id`
- `thread_id`
- `action`
- `candidate_category`
- `candidate_content`
- `target_memory_id`
- `result_memory_id`
- `reason`
- `created_at`

语义说明：

- `target_memory_id`: 被命中的旧记忆；`add` / `ignore` 时允许为空
- `result_memory_id`: 最终生效的记忆；`ignore` 时允许为空
- `reason`: 第一版先保存规则侧原因码，避免放模糊自然语言

## Domain Rules

### add

执行条件：

- 没有找到足够相似的 `active` 记忆

执行结果：

- 插入一条新记忆
- `status='active'`
- `reinforcement_count=0`

### reinforce

执行条件：

- 找到相似 `active` 记忆
- 新旧内容语义上没有明显冲突
- 旧记忆的重要性或表述已经足够稳定

执行结果：

- 不新增记忆
- 命中的旧记忆 `reinforcement_count + 1`
- 可选地把 importance 向上收敛，但 MVP 建议只在上限 5 内加 1
- 更新 `last_reinforced_at`

### revise

执行条件：

- 找到相似 `active` 记忆
- 新内容和旧内容主题相同，但表述更完整、更准确或出现明确纠正信号

执行结果：

- 旧记忆改为 `superseded`
- 插入一条新记忆，`status='active'`
- 新记忆的 `supersedes_memory_id` 指向旧记忆

### ignore

执行条件：

- 候选内容为空、过短、敏感或明显不值得长期保存
- 或者与现有记忆相比没有提供新信息，也不值得强化

执行结果：

- 不新增也不修改记忆
- 只记录 audit

## Decision Heuristics for MVP

第一版不用额外 LLM 决策器，优先采用规则判断，避免系统复杂度陡增。

建议顺序：

1. 过滤空内容、敏感内容、极短内容
2. 在同 user 下按 category 检索相关 `active` 记忆
3. 如果没有相似候选，执行 `add`
4. 如果有高相似候选：
   - 出现明显修正词，如“不是…而是… / 改成 / 更准确地说 / 实际上 / 以后以…为准”，优先尝试 `revise`
   - 如果只是重复确认或轻微同义复述，执行 `reinforce`
   - 如果新内容没有增加有效信息，执行 `ignore`

这里的“明显修正词”应被封装到独立 helper 中，而不是散落到 store 代码里。

## Repository and Runtime Boundaries

### Semantic memory repository

需要新增最小读写能力：

- 只查询 `active` 记忆
- 更新单条记忆的状态
- 更新强化计数和时间
- 插入带 `supersedes_memory_id` 的新记忆

### Semantic store

新增一个显式的“演化入口”，例如：

- `learn_memory(...)`
- 或 `evolve_memory(...)`

它负责：

- 找候选旧记忆
- 做动作判断
- 调 repository 完成写入
- 返回结构化结果，供 audit 和测试使用

当前的 `add_memory()` 保留，用于低层直接插入；学习路径逐步转到演化入口。

### Agent / LangGraph learn node

classic 和 langgraph 两条学习路径都改为调用同一套长期记忆演化入口，而不是直接 `add_memory()`。

目标是：

- orchestration 差异继续留在 runtime 层
- 记忆演化规则只有一份

## CLI Surface

MVP 不新增大命令面，但要补一个最小查看入口。

建议：

- `/learning` 保持现状
- `/thread` 审计时间线增加 memory evolution 事件
- 视实现成本，可新增 `/memory-log [thread=<id>] [limit=<n>]`

如果这一步会把 CLI 范围拉大，MVP 可以先只把演化事件接进 `/thread`，后续再决定是否补单独命令。

## Testing Strategy

### Unit tests

新增或扩展这些测试面：

- 候选记忆被判定为 `add`
- 重复确认触发 `reinforce`
- 明确纠正触发 `revise`
- 低价值内容触发 `ignore`
- `superseded` 记忆不会被默认检索
- `reinforcement_count` 和 `last_reinforced_at` 正确更新

### Store / repository tests

验证：

- schema migration 可在老数据库上补齐新列和新表
- active / superseded 状态读写正确
- revision 链接字段正确保存
- audit 记录字段完整

### Integration tests

验证：

- classic `ConversationalAgent._learn_from_turn()` 经过演化入口
- `LangGraphAgent._learn_node()` 经过同一演化入口
- `/thread` 能显示演化事件

## Acceptance Criteria

当这个 MVP 完成时，应满足：

1. agent 学习新长期记忆时，不再只会直接追加
2. 至少支持 `add / reinforce / revise / ignore` 四种动作
3. 长期记忆可以区分 `active` 和 `superseded`
4. 默认检索只使用 `active` 记忆
5. classic 和 langgraph 的学习路径共享同一套记忆演化规则
6. 每次记忆演化都有可查询的审计记录
7. 现有记忆治理命令和测试仍然可用

## Suggested Implementation Slice

为了保持最小风险，建议按这一顺序落地：

1. schema 和 dataclass 补齐 `active/superseded` 与 evolution audit
2. semantic repository / store 增加 `reinforce`、`revise` 所需写接口
3. 实现 memory evolution decision helper
4. classic 和 langgraph 学习路径切到统一的演化入口
5. `/thread` 展示演化审计
6. 补齐测试和文档

## Open Decisions

这版 spec 里还保留两个故意延后的决定：

1. `reinforce` 是否每次都提高 importance，还是只增加 `reinforcement_count`
2. 是否需要单独的 `/memory-log` CLI 命令

这两个点不会阻塞 MVP 核心能力，可以在实现时做保守默认：

- importance 上调采用 `min(5, importance + 1)`
- CLI 先只接入 `/thread`
