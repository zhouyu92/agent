# Roadmap

## 当前状态

这个仓库已经不再是一个只有基础对话能力的原型，而是进入了“可持续演进”的第一阶段末尾。

目前已经完成的主线包括：

- 基于阿里云百炼 OpenAI 兼容接口接入 `qwen3.7-max`
- 用 SQLite 落长期记忆、agent profile、审计事件和对话转录
- 建立 classic / langgraph 双后端运行路径
- 为 LangGraph 路径补上 runtime、routing policy、thread inspection
- 落地 memory evolution MVP，支持 `add / reinforce / revise / ignore`
- 为长期记忆引入 `active / superseded` 状态，默认只让 `active` 记忆进入回复上下文
- 为学习、检索、去重、记忆演化补上 CLI 可观测能力和审计轨迹

一句话说，当前基座已经具备：

- 短期记忆
- 长期记忆
- 记忆演化
- 基本治理
- 运行时观测

还没完成的，不是“能不能跑”，而是“能不能稳定地学对、管住、持续进化”。

## Phase 1：打磨 Memory Evolution MVP

目标：先把“学什么、怎么改、何时忽略”这套规则做稳。

建议聚焦：

- 收紧 `add / reinforce / revise / ignore` 的判定边界
- 增加冲突偏好、信息过时、同义重复、弱信号输入的测试
- 让 routing policy 和 memory evolution 的职责边界更清楚
- 补强 CLI 查询视图，方便快速审计某一线程或某一类记忆

完成标志：

- 常见错误学习场景都有测试覆盖
- 误写入和重复写入明显下降
- 人工查看 `/thread`、`/learning`、`/memories` 时，能快速解释每次学习结果

## Phase 2：长期记忆治理

目标：让长期记忆不是“越存越多”，而是“越存越干净”。

建议聚焦：

- 为记忆补充更明确的质量信号，比如来源、最近确认时间、冲突状态
- 增加记忆归档、压缩、淘汰或人工确认机制
- 形成一套可重复执行的 memory hygiene 流程
- 继续保持 SQLite 为主，不急着引入新的生产存储

完成标志：

- 记忆库能区分高价值记忆和低价值噪声
- 过时或被修正的记忆不会继续污染默认回复上下文
- 日常治理可以通过现有 CLI 完成，不需要临时写脚本排查

## Phase 3：自我学习回路

目标：让 agent 学到的不只是“用户说过什么”，还包括“应该怎样更好地互动”。

建议聚焦：

- 从多轮对话中提炼稳定偏好、关系感知和目标变化
- 建立 `episode -> reflection -> durable memory` 的学习链路
- 把“策略经验”与“事实记忆”区分存放，避免混杂
- 让 agent profile 的更新更可解释、更克制

完成标志：

- agent 能从连续交流中表现出更稳定的沟通风格适配
- 新学到的内容不只是事实条目，还包括互动策略和边界理解
- profile 和 memory 的更新都能被审计解释

## Phase 4：LangGraph Phase 2

目标：在现有建模稳定后，再把生产级 orchestration 做清楚。

建议聚焦：

- 继续落实 `Checkpoint / ThreadStateStore / LongTermStore` 的边界
- 收紧 bootstrap 装配入口，避免 graph 私有细节泄漏到外围
- 让 classic / langgraph 的差异主要留在 orchestration 层
- 为后续工具调用、记忆整理节点、失败恢复预留结构

完成标志：

- `build_runtime()` 成为唯一清晰的主装配入口
- LangGraph 负责图内状态与流程，长期记忆继续独立治理
- CLI 查询路径不依赖 backend 内部实现细节

## 当前建议

现在最值得做的，不是立刻把所有能力迁到 LangGraph，而是继续完成 Phase 1 和 Phase 2。

原因很简单：

- 记忆模型一旦没收稳，换框架只会把问题搬到更复杂的地方
- 当前仓库已经有足够好的 LangGraph 落点，不需要抢跑
- 先把 memory evolution 和 long-term governance 做扎实，第二阶段迁移会更顺
