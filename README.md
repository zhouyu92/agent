# Agent

一个本地命令行 agent MVP：使用阿里云百炼 OpenAI 兼容接口调用 `qwen3.7-max`，并用 SQLite 保存长期记忆与自我画像。

## 目标

- 像真人一样保持连续、有温度的交流，但不伪装成人类。
- 从对话中提炼长期有用的信息，持续更新记忆。
- 维护一个可演进的 agent profile，用于调整沟通风格和边界。
- API 凭据只通过环境变量读取，不写入代码或仓库。

## 配置

复制 `.env.example` 为 `.env`，填入你的配置：

```env
DASHSCOPE_API_KEY=sk-...
DASHSCOPE_WORKSPACE_ID=your-workspace-id
DASHSCOPE_MODEL=qwen3.7-max
AGENT_USER_ID=default
AGENT_BACKEND=classic
AGENT_CHECKPOINT_DB=data/checkpoints.db
```

也可以直接设置完整地址：

```env
DASHSCOPE_BASE_URL=https://your-workspace-id.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
```

注意：如果 API key 已经在聊天、日志或公开位置暴露，建议立即轮换。

如果要切到第二阶段的 LangGraph 后端，把 `AGENT_BACKEND` 改成 `langgraph`。这时线程内短期记忆会走 LangGraph checkpoint，长期记忆会先通过 `LongTermStore` 适配层接入现有 SQLite store。

## 安装与运行

```bash
python -m pip install -e ".[dev]"
agent-chat
```

诊断配置和本地记忆数据库：

```bash
agent-doctor
```

在填好有效 `DASHSCOPE_API_KEY` 和 `DASHSCOPE_WORKSPACE_ID` 后，可额外做一次模型连通性检查：

```bash
agent-doctor --online
```

退出聊天：

```text
/exit
```

常用命令：

```text
/profile           查看当前 agent 自我画像
/memories          查看最近保存的长期记忆
/memories 回答偏好  检索相关长期记忆
/memories category=preference
/memories importance=4
/memories category=preference importance=4 回答偏好
/forget 12         删除当前用户下 id 为 12 的长期记忆
/dedupe-memories
/dedupe-memories thread=t10
/dedupe-log
/dedupe-log thread=t10
/dedupe-log thread=t10 limit=5
/learning          查看最近学习事件
/routing           查看最近的检索/学习路由决策
/routing thread=t10
/routing learn=false
/routing recall_turn
/thread t10
```

`/thread <thread_id>` 会把同一线程里的用户消息、agent 回复、routing 决策、retrieval 摘要、memory evolution 事件和 learning 事件按时间顺序合并展示。

## 记忆机制

- 短期记忆：同一个 `thread_id` 的最近若干轮消息会注入上下文。
- 长期记忆：对话后由模型提炼 JSON 记忆，按 `AGENT_USER_ID` 保存到 SQLite。
- 检索：回复前只检索当前 `AGENT_USER_ID` 的相关长期记忆，并注入 system prompt。
- 长期记忆演化：学习阶段不会只做追加；系统会对候选记忆执行 `add / reinforce / revise / ignore`。
- 默认检索只使用 `active` 记忆；被修正的 `superseded` 记忆保留用于审计，不进入默认回复上下文。
- 纠错：`/memories` 会显示记忆 id，并支持 `category=`、`importance=` 过滤；可用 `/forget <id>` 删除错误记忆，或用 `/dedupe-memories [thread=<id>]` 清理重复记忆并输出被删除的 id 列表。`/dedupe-log [thread=<id>] [limit=<n>]` 可查看最近去重记录。
- 自我进化：`agent_profile` 保存身份、风格和边界，只在学习步骤明确返回更新时变化。
- 学习事件：每轮学习会记录写入记忆数量和 profile 更新字段，便于用 `/learning` 审计变化。
- LangGraph 后端：通过 `AGENT_BACKEND=langgraph` 启用，短期记忆走 `AGENT_CHECKPOINT_DB`，长期记忆继续复用 SQLite。

## 记忆治理

一个常见的人工治理流程可以这样走：

```text
/memories
/memories category=preference
/memories importance=4 回答偏好
/forget 12
/dedupe-memories
/dedupe-memories thread=t10
/dedupe-log thread=t10
/dedupe-log thread=t10 limit=5
```

- 先用 `/memories` 看最近写入的长期记忆。
- 再用 `category=`、`importance=` 和查询词缩小范围，检查某一类记忆是否准确。
- 发现错误记忆时，用 `/forget <id>` 删除单条。
- 发现同类表述重复时，用 `/dedupe-memories` 清理重复项，并核对返回的删除 id。
- 如果这次治理只想归属到某个线程，可以用 `thread=<id>` 执行去重，再用 `/dedupe-log thread=<id> limit=<n>` 回看该线程下最近若干条去重记录。

示例输出：

```text
/dedupe-log thread=t10 limit=2
- [2026-07-04T12:00:00+00:00] removed=2 ids=7,9
- [2026-07-04T11:42:10+00:00] removed=1 ids=4
```

## LangGraph 架构（第二阶段）

当前 LangGraph 后端采用一个最小但职责清晰的三节点流程：

- `retrieve`：根据当前用户输入检索长期记忆，并把检索结果写入 graph state。
- `respond`：读取 checkpoint 中的线程消息和 `retrieve` 产出的记忆，生成回复。
- `learn`：从本轮对话中提炼新的长期记忆和 profile 更新，并写回 SQLite。

此外，`respond` 之后现在有一个最小条件路由：如果当前用户输入看起来像敏感凭据、命令或明显低信息量的寒暄，graph 会直接结束，不再进入学习节点。
同样地，`retrieve` 前也会做一个最小判断：明显低价值的轮次会跳过长期记忆检索，直接进入回复节点。
这两套判断现在已经分开: 像“你还记得我的回答偏好吗？”这类回忆型问题会触发检索，但不会被当成新的学习样本写回。
这些判断现在收敛在独立的 routing policy 层里，LangGraph 负责执行节点和路由，policy 负责回答“这一轮值不值得检索/学习”。
目前 policy 还会给出原因码，并把决策结果放进 graph state，便于后面做审计、观测和策略调优。
同时，每轮的 routing 决策现在也会落到独立的审计记录里，这样即使某一轮没有进入学习节点，也能追踪“为什么没检索 / 为什么没学习”。

这样做的目的是先把短期记忆、长期记忆、回复生成、学习回写这几层边界拆开，后续再继续接入条件路由、工具调用或记忆整理节点。

## 测试

```bash
python -m pytest
```
