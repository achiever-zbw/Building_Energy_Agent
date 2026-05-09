"""可编排响应的 LLM Provider，用于 Agent / Consolidator 单测。"""
from __future__ import annotations

from collections import deque
from typing import Any

from backend.agent.providers.base import LLMProvider
from backend.agent.schemas import LLMResponse


class ScriptedProvider(LLMProvider):
    """按顺序弹出预设 ``LLMResponse``；每次 ``chat`` 消耗队列头部一条。"""

    def __init__(self, responses: list[LLMResponse]) -> None:
        super().__init__(api_key="sk-mock", api_base="http://mock.local")
        self._queue: deque[LLMResponse] = deque(responses)
        self.chat_invocations: list[dict[str, Any]] = []

    def push(self, response: LLMResponse) -> None:
        self._queue.append(response)

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.2,
        reasoning_effort: str | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.chat_invocations.append(
            {
                "messages_len": len(messages),
                "tools_len": len(tools or []),
                "model": model,
            }
        )
        if self._queue:
            return self._queue.popleft()
        return LLMResponse(
            content="[mock provider: no scripted responses left]",
            finish_reason="stop",
        )

    def get_default_model(self) -> str:
        return "mock-model"
