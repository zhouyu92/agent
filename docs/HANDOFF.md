# Handoff

## Current state

- Local branch: `main`
- Local commit: `8956758 Initial self-learning agent MVP`
- Remote: `https://github.com/zhouyu92/agent.git`
- Remote push status: not pushed yet

The project currently provides:

- `agent-chat`: CLI chat agent.
- `agent-doctor`: offline config/database checks and optional online model ping.
- SQLite-backed short-term messages, long-term memories, learning events, and agent profile.
- Per-user memory isolation via `AGENT_USER_ID`.
- Sensitive value redaction before message/event persistence.
- Memory deletion with `/forget <memory_id>`.

## Verification

Last verified locally with:

```powershell
python -m pytest
python -m compileall src tests
rg "paste-sensitive-key-fragments-here-before-release" -n .
```

Expected result:

- All tests pass.
- Source and tests compile.
- No real API key is present in tracked files.

## Push to GitHub

The local repository is ready to push, but HTTPS authentication failed because no valid GitHub token is configured.

After configuring GitHub credentials, run:

```powershell
git push origin main
```

Two common options:

1. Use HTTPS with a GitHub Personal Access Token as the password when Git prompts.
2. Switch the remote to SSH after adding an SSH key to GitHub:

```powershell
git remote set-url origin git@github.com:zhouyu92/agent.git
git push origin main
```

Do not commit `.env`, real API keys, SQLite databases, or cache directories.

## Runtime setup

Copy `.env.example` to `.env` and fill in local secrets:

```env
DASHSCOPE_API_KEY=your-rotated-key
DASHSCOPE_BASE_URL=https://llm-8oybi655nr8jwsyh.cn-beijing.maas.aliyuncs.com/compatible-mode/v1
DASHSCOPE_MODEL=qwen3.7-max
AGENT_USER_ID=default
```

Then run:

```powershell
python -m pip install -e ".[dev]"
agent-doctor
agent-doctor --online
agent-chat
```

The API keys shared during development were exposed in conversation and should be rotated before long-term use.
