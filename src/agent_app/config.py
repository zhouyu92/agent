from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class AgentConfig:
    api_key: str
    base_url: str
    model: str = "qwen3.7-max"
    memory_db_path: Path = Path("data/agent.db")
    max_recent_turns: int = 8
    user_id: str = "default"

    @classmethod
    def from_env(cls) -> "AgentConfig":
        load_dotenv()

        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY is required.")

        base_url = os.getenv("DASHSCOPE_BASE_URL")
        workspace_id = os.getenv("DASHSCOPE_WORKSPACE_ID")
        if not base_url:
            if not workspace_id:
                raise ValueError("Set DASHSCOPE_BASE_URL or DASHSCOPE_WORKSPACE_ID.")
            base_url = f"https://{workspace_id}.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"

        return cls(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            model=os.getenv("DASHSCOPE_MODEL", "qwen3.7-max"),
            memory_db_path=Path(os.getenv("AGENT_MEMORY_DB", "data/agent.db")),
            max_recent_turns=int(os.getenv("AGENT_MAX_RECENT_TURNS", "8")),
            user_id=os.getenv("AGENT_USER_ID", "default"),
        )
