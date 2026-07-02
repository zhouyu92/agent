from __future__ import annotations

from collections.abc import Sequence

from openai import OpenAI

from .config import AgentConfig


class QwenClient:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def chat(self, messages: Sequence[dict[str, str]], temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=list(messages),
            temperature=temperature,
        )
        content = response.choices[0].message.content
        return content or ""

    def ping(self) -> str:
        return self.chat([{"role": "user", "content": "ping"}], temperature=0.0)
