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
```

也可以直接设置完整地址：

```env
DASHSCOPE_BASE_URL=https://your-workspace-id.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
```

注意：如果 API key 已经在聊天、日志或公开位置暴露，建议立即轮换。

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
/forget 12         删除当前用户下 id 为 12 的长期记忆
/learning          查看最近学习事件
```

## 记忆机制

- 短期记忆：同一个 `thread_id` 的最近若干轮消息会注入上下文。
- 长期记忆：对话后由模型提炼 JSON 记忆，按 `AGENT_USER_ID` 保存到 SQLite。
- 检索：回复前只检索当前 `AGENT_USER_ID` 的相关长期记忆，并注入 system prompt。
- 纠错：`/memories` 会显示记忆 id，可用 `/forget <id>` 删除当前用户下的错误记忆。
- 自我进化：`agent_profile` 保存身份、风格和边界，只在学习步骤明确返回更新时变化。
- 学习事件：每轮学习会记录写入记忆数量和 profile 更新字段，便于用 `/learning` 审计变化。

## 测试

```bash
python -m pytest
```
