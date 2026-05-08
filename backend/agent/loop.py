"""Agent loop：串联 ContextBuilder → ToolRegistry → AgentRunner 的请求生命周期（对齐 nanobot 思路）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

from backend.agent.consolidator import Consolidator
from backend.agent.context import ContextBuilder
from backend.agent.hook import AgentHook, AgentHookContext, CompositeHook
from backend.agent.providers.base import LLMProvider
from backend.agent.runner import AgentRunner
from backend.agent.schemas import AgentRunResult, AgentRunSpec
from backend.agent.message_dump import log_agent_messages_block
from backend.agent.tools.registry import ToolRegistry
from backend.utils.helpers import strip_think


def _log_history_before_model(chat_id: str, hist: list[dict[str, Any]], user_message: str) -> None:
    """本轮请求模型前：仅含过往轮的 history（无 system、无本条 user）；本条 user 单独打印。"""
    log_agent_messages_block(
        f"[run_turn prior history only] chat_id={chat_id}",
        hist,
        iteration=None,
    )
    um = user_message if len(user_message) <= 400_000 else user_message[:400_000] + "\n... (truncated)"
    logger.info(f"[run_turn current user message] chat_id={chat_id}\n{um}")


class _LoopHook(AgentHook):
    """主循环内置 Hook：迭代序号、流式增量、工具前日志与用量日志。"""

    def __init__(
        self,
        agent_loop: AgentLoop,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        *,
        chat_id: str | None = None,
    ) -> None:
        super().__init__(reraise=True)
        self._loop = agent_loop
        self._on_progress = on_progress
        self._on_stream = on_stream
        self._on_stream_end = on_stream_end
        self._chat_id = chat_id
        self._stream_buf = ""

    def wants_streaming(self) -> bool:
        return self._on_stream is not None

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        prev_clean = strip_think(self._stream_buf)
        self._stream_buf += delta
        new_clean = strip_think(self._stream_buf)
        incremental = new_clean[len(prev_clean) :]
        if incremental and self._on_stream:
            await self._on_stream(incremental)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        if self._on_stream_end:
            await self._on_stream_end(resuming=resuming)
        self._stream_buf = ""

    async def before_iteration(self, context: AgentHookContext) -> None:
        """ 同步为当前 ReAct 轮次，方便外界监控 """
        self._loop._current_iteration = context.iteration

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        """ 在模型已经返回 response 并且决定执行工具 """
        if self._on_progress:
            if not self._on_stream and not context.streamed_content:
                thought = self._loop._strip_think(
                    context.response.content if context.response else None,
                )
                if thought:
                    await self._on_progress(thought)
            hint = self._loop._strip_think(self._loop._tool_hint(context.tool_calls))
            if hint:
                await self._on_progress(hint)
        for tc in context.tool_calls:
            args_str = json.dumps(tc.arguments, ensure_ascii=False)
            logger.info("Tool call: {}({})", tc.name, args_str[:200])

    async def after_iteration(self, context: AgentHookContext) -> None:
        """ 一次迭代完成后，打日志 """
        u = context.usage or {}
        logger.debug(
            "LLM usage: prompt={} completion={} cached={}",
            u.get("prompt_tokens", 0),
            u.get("completion_tokens", 0),
            u.get("cached_tokens", 0),
        )

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        """ runner 判定结束后进行 final hook ，对内容进行去 think，避免思考标签 """
        return self._loop._strip_think(content)


class AgentLoop:
    """
    单次或并发请求外的编排层：负责
    1. 用 ContextBuilder 组装 system + 历史 + 当前用户消息；
    2. 持有 ToolRegistry 与 AgentRunner；
    3. 通过 Hook 向前端/日志透出进度（可选流式）。
    """

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        *,
        model: str | None = None,
        max_iterations: int = 15,
        context_window_tokens: int | None = None,
        context_block_limit: int | None = None,
        max_tool_result_chars: int = 32_000,
        timezone: str | None = None,
        tools: ToolRegistry | None = None,
        hooks: list[AgentHook] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        fail_on_tool_error: bool = False,
        concurrent_tools: bool = True,
        enable_consolidation: bool = True,
        consolidation_ratio: float = 0.5,
    ) -> None:
        self.provider = provider
        self.workspace = Path(workspace)
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.context_window_tokens = (
            context_window_tokens if context_window_tokens is not None else 128_000
        )
        self.context_block_limit = context_block_limit
        self.max_tool_result_chars = max_tool_result_chars
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.fail_on_tool_error = fail_on_tool_error
        self.concurrent_tools = concurrent_tools
        self.enable_consolidation = enable_consolidation

        self.context = ContextBuilder(self.workspace, timezone=timezone)
        self.tools = tools if tools is not None else ToolRegistry()
        # Agent 核心 ReAct 循环运行机制
        self.runner = AgentRunner(provider)
        self._extra_hooks: list[AgentHook] = list(hooks or [])

        if enable_consolidation:
            self.consolidator = Consolidator(
                store=self.context.memory,
                provider=provider,
                model=self.model,
                context_window_tokens=self.context_window_tokens,
                context_builder=self.context,
                tools=self.tools,
                consolidation_ratio=consolidation_ratio,
            )
        else:
            self.consolidator = None

        self._last_usage: dict[str, int] = {}
        self._current_iteration: int = 0

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        if not text:
            return None
        cleaned = strip_think(text)
        return cleaned if cleaned else None

    @staticmethod
    def _tool_hint(tool_calls: list[Any]) -> str:
        if not tool_calls:
            return ""
        parts: list[str] = []
        for tc in tool_calls:
            name = getattr(tc, "name", "") or ""
            parts.append(f"{name}(…)")
        return " · ".join(parts)

    def _compose_hook(
        self,
        loop_hook: _LoopHook,
    ) -> AgentHook:
        if self._extra_hooks:
            return CompositeHook([loop_hook, *self._extra_hooks])
        return loop_hook

    def _sync_consolidator_runtime(self) -> None:
        if self.consolidator is None:
            return
        self.consolidator.model = self.model
        self.consolidator.context_window_tokens = self.context_window_tokens
        self.consolidator.provider = self.provider
        self.consolidator.max_completion_tokens = int(
            getattr(getattr(self.provider, "generation", None), "max_tokens", 4096),
        )

    async def _run_agent_loop(
        self,
        initial_messages: list[dict[str, Any]],
        *,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        chat_id: str | None = None,
        checkpoint_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        error_message: str | None = None,
        llm_timeout_s: float | None = None,
    ) -> AgentRunResult:
        loop_hook = _LoopHook(
            self,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            chat_id=chat_id,
        )
        hook = self._compose_hook(loop_hook)

        spec_kw: dict[str, Any] = {
            "initial_messages": initial_messages,
            "tools": self.tools,
            "model": self.model,
            "max_iterations": self.max_iterations,
            "max_tool_result_chars": self.max_tool_result_chars,
            "temperature": self.temperature,
            "workspace": self.workspace,
            "max_tokens": self.max_tokens,
            "context_block_limit": self.context_block_limit,
            "reasoning_effort": self.reasoning_effort,
            "hook": hook,
            "fail_on_tool_error": self.fail_on_tool_error,
            "context_window_tokens": self.context_window_tokens,
            "checkpoint_callback": checkpoint_callback,
            "concurrent_tools": self.concurrent_tools,
            "llm_timeout_s": llm_timeout_s,
        }
        if error_message is not None:
            spec_kw["error_message"] = error_message
        spec = AgentRunSpec(**spec_kw)
        result = await self.runner.run(spec)
        self._last_usage = dict(result.usage or {})
        if result.stop_reason == "max_iterations":
            logger.warning("达到 max_iterations={}，已停止本轮 ReAct", self.max_iterations)
            if on_stream and on_stream_end and result.final_content:
                await on_stream(result.final_content)
                await on_stream_end(resuming=False)
        return result

    async def run_turn(
        self,
        user_message: str,
        *,
        history: list[dict[str, Any]] | None = None,
        chat_id: str | None = None,
        session_summary: str | None = None,
        persist_history: bool = False,
        on_progress: Callable[..., Awaitable[None]] | None = None,
        on_stream: Callable[[str], Awaitable[None]] | None = None,
        on_stream_end: Callable[..., Awaitable[None]] | None = None,
        checkpoint_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
        error_message: str | None = None,
        llm_timeout_s: float | None = None,
        consolidate: bool = True,
        session_key: str | None = None,
    ) -> AgentRunResult:
        """
        处理一轮用户输入：先可选执行 Consolidator 压缩 history，再 build_message → AgentRunner.run。
        persist_history 为 True 时，在 ``memory/history.jsonl`` 中追加用户句与助手最终回复（纯文本摘要）。
        consolidate=False 可跳过本轮压缩；session_key 用于压缩锁（默认 chat_id 或 ``default``）。
        """
        self._sync_consolidator_runtime()
        hist = list(history or [])
        sk = session_key or chat_id or "default"
        rollup: str | None = None
        if consolidate and self.consolidator is not None:
            rollup = await self.consolidator.maybe_consolidate_history(
                hist,
                chat_id=chat_id,
                session_summary=session_summary,
                session_key=sk,
            )
        if rollup and session_summary:
            effective_summary = f"{rollup}\n\n---\n\n{session_summary}"
        else:
            effective_summary = rollup or session_summary

        # 调试：本轮请求模型前，会话 history（不含稍后经 build_message 注入的 system）
        _log_history_before_model(chat_id or sk, hist, user_message)

        # 构建初始消息，整合历史信息、用户的输入等
        initial = self.context.build_message(
            hist,
            user_message,
            current_role="user",
            chat_id=chat_id,
            session_summary=effective_summary,
        )
        # info 出来
        log_agent_messages_block(
            "[run_turn initial_messages full — system + history + current user]",
            initial,
            iteration=None,
        )
        # 执行 agent
        result = await self._run_agent_loop(
            initial,
            on_progress=on_progress,
            on_stream=on_stream,
            on_stream_end=on_stream_end,
            chat_id=chat_id,
            checkpoint_callback=checkpoint_callback,
            error_message=error_message,
            llm_timeout_s=llm_timeout_s,
        )

        log_agent_messages_block(
            "[run_turn result.messages — full trajectory after run]",
            result.messages,
            iteration=None,
        )

        # 完善历史信息
        if persist_history:
            try:
                mem = self.context.memory
                # 把用户的信息先加入到 history
                mem.append_history(user_message)
                if result.final_content:
                    # 如果存在 Agent 的返回结果，把结果也加入到历史信息
                    mem.append_history(result.final_content)
            except Exception:
                logger.exception("写入 history.jsonl 失败，已忽略")
        return result

    @property
    def last_usage(self) -> dict[str, int]:
        return dict(self._last_usage)
