"""按 token 预算触发：将过期前缀摘要写入 history.jsonl，并从传入的 history 中删除（对齐 nanobot Consolidator 思路，无 SessionManager）。"""

from __future__ import annotations

import asyncio
import json
import weakref
from typing import Any

import tiktoken
from loguru import logger

from backend.agent.context import ContextBuilder
from backend.agent.memory import MemoryStore
from backend.agent.providers.base import LLMProvider
from backend.agent.tools.registry import ToolRegistry
from backend.utils.helpers import (
    estimate_message_tokens,
    estimate_prompt_tokens_chain,
    find_legal_message_start,
    truncate_text,
)

_MAX_CONSOLIDATION_ROUNDS = 5
_SAFETY_BUFFER = 1024
_ARCHIVE_SUMMARY_MAX_CHARS = 8000
_RAW_ARCHIVE_MAX_CHARS = 16_000

_ARCHIVE_SYSTEM_PROMPT = (
    "你是对话压缩助手。将给定消息摘要为可供后续轮次参考的要点，使用与用户相同的语言。"
    "只输出要点列表或短段落：事实、决策、未完成任务；不要角色扮演或重复闲聊。"
)


class Consolidator:
    """ 记忆压缩
    在对话历史过长时自动触发摘要压缩，确保对话不超过模型的上下文窗口
    """

    def __init__(
        self,
        # 持久化存储
        store: MemoryStore,
        provider: LLMProvider,
        model: str,
        context_window_tokens: int,
        context_builder: ContextBuilder,
        tools: ToolRegistry,
        *,
        max_completion_tokens: int | None = None,
        # 触发压缩阈值
        consolidation_ratio: float = 0.5,
        enabled: bool = True,
    ) -> None:
        self.store = store
        self.provider = provider
        self.model = model
        self.context_window_tokens = context_window_tokens
        self._context = context_builder
        self._tools = tools
        self.consolidation_ratio = consolidation_ratio
        self.enabled = enabled
        self.max_completion_tokens = max_completion_tokens or getattr(
            getattr(provider, "generation", None), "max_tokens", 4096
        )
        try:
            self.max_completion_tokens = int(self.max_completion_tokens)
        except (TypeError, ValueError):
            self.max_completion_tokens = 4096

        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    def set_provider(
        self,
        provider: LLMProvider,
        model: str,
        context_window_tokens: int,
    ) -> None:
        self.provider = provider
        self.model = model
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = int(
            getattr(getattr(provider, "generation", None), "max_tokens", 4096)
        )

    def _lock(self, session_key: str) -> asyncio.Lock:
        return self._locks.setdefault(session_key, asyncio.Lock())

    @property
    def _input_token_budget(self) -> int:
        """
        输入给 model 的预算 = 最大窗口限制 - 最大输出预留 - 安全缓冲
        """
        return (
            self.context_window_tokens
            - max(1, self.max_completion_tokens)
            - _SAFETY_BUFFER
        )

    def _truncate_to_token_budget(self, text: str) -> str:
        """ 截断到 token 预算
        """
        # 获取 message 的 token 预算
        budget = self._input_token_budget
        if budget <= 0:
            # 截断字符串
            return truncate_text(text, _RAW_ARCHIVE_MAX_CHARS)
        try:
            # token 级的截断，str 转化为 token 列表
            enc = tiktoken.get_encoding("cl100k_base")
            tokens = enc.encode(text)
            # token 长度 <= 预算，不需要截断
            if len(tokens) <= budget:
                return text
            # 保留前 budget 个 token ，并转化为字符串并拼接 (truncated) 进行截断说明
            return enc.decode(tokens[:budget]) + "\n... (truncated)"
        except Exception:
            return truncate_text(text, max(256, budget * 4))

    @staticmethod
    def _format_chunk(messages: list[dict[str, Any]]) -> str:
        """ 将消息列表拼接成 str 用来给 LLM 进行摘要压缩 """
        lines: list[str] = []
        for m in messages:
            role = str(m.get("role", "?"))
            content = m.get("content")
            if isinstance(content, list):
                content = json.dumps(content, ensure_ascii=False)
            elif content is None:
                content = ""
            if m.get("tool_calls"):
                content = f"{content}\n[tool_calls: {len(m['tool_calls'])}]"
            lines.append(f"{role.upper()}: {content}")
        return "\n".join(lines)

    def _estimate_prompt(
        self,
        history: list[dict[str, Any]],
        *,
        chat_id: str | None,
        session_summary: str | None,
    ) -> tuple[int, str]:
        """ 估算当前对话历史在完整模型请求中会消耗多少 token """
        # 构建完整的 message , 塞入一个假的 current_message
        probe_messages = self._context.build_message(
            list(history),
            "[token-probe]",
            current_role="user",
            chat_id=chat_id,
            session_summary=session_summary,
        )
        # 计算 token 消耗量
        return estimate_prompt_tokens_chain(
            self.provider,
            self.model,
            probe_messages,
            self._tools.get_definitions(),
        )

    @staticmethod
    def _pick_boundary(history: list[dict[str, Any]], tokens_to_remove: int) -> int | None:
        """
        Args:
            history (list[dict[str, Any]]) 当前对话历史消息列表
            tokens_to_remove int 需要删除的 token 数量
        Returns:
            idx int 索引，表示应该删除 history[:idx] 的消息
        """
        if not history or tokens_to_remove <= 0:
            return None
        # 累计已扫描消息的总 token
        removed = 0
        candidate: int | None = None
        for idx, msg in enumerate(history):
            if idx > 0 and msg.get("role") == "user":
                # idx = 0 不删去，没有意义
                # user 的信息，删去，更新游标
                candidate = idx
                # 删除的总和足够了
                if removed >= tokens_to_remove:
                    return candidate
            # 累计删除的 token
            removed += estimate_message_tokens(msg)
        return candidate

    @staticmethod
    def _repair_orphans(history: list[dict[str, Any]]) -> None:
        start = find_legal_message_start(history)
        if start > 0:
            del history[:start]

    async def archive(self, messages: list[dict[str, Any]]) -> str | None:
        """ 将一组 message 通过 LLM 压缩成摘要吗，并将摘要写入 MemoryStore """
        if not messages:
            return None
        try:
            # 拼接成字符串
            formatted = self._format_chunk(messages)
            # 截断到输入 token 的预算以内
            formatted = self._truncate_to_token_budget(formatted)
            # 把需求给 LLM ，压缩摘要
            response = await self.provider.chat_with_retry(
                model=self.model,
                messages=[
                    {"role": "system", "content": _ARCHIVE_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"请压缩以下对话片段：\n\n{formatted}",
                    },
                ],
                tools=None,
                tool_choice=None,
                max_tokens=min(2048, max(256, self.max_completion_tokens)),
                temperature=0.2,
            )
            if response.finish_reason == "error":
                raise RuntimeError(response.content or "LLM error")

            # 获取压缩结果
            summary = (response.content or "").strip() or "[no summary]"
            # 在 history 中添加压缩后的信息
            self.store.append_history(
                f"[Consolidated {len(messages)} messages]\n{summary}",
            )
            return truncate_text(summary, _ARCHIVE_SUMMARY_MAX_CHARS)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Consolidation LLM 失败，改用 RAW 写入 history.jsonl")
            raw = truncate_text(
                self._format_chunk(messages),
                _RAW_ARCHIVE_MAX_CHARS,
            )
            self.store.append_history(
                f"[RAW consolidation] {len(messages)} messages\n{raw}",
            )
            return None

    async def maybe_consolidate_history(
        self,
        history: list[dict[str, Any]],
        *,
        chat_id: str | None,
        session_summary: str | None,
        session_key: str,
    ) -> str | None:
        """若估算 prompt 超过目标预算，则反复归档前缀直至达标或无路可切。

        会直接 原地修改 history （删除已归档消息并修复 tool 孤儿）。

        返回：本轮新生成的摘要片段拼接串，供与 ``session_summary`` 合并注入 Runtime。

        完整流程：
            1. 触发：用户发送消息后，构建 message 列表，会查看是否达到压缩阈值
            2. 多轮压缩循环
                （1）估算当前 history 的 token 消耗数值，< budget 就不需要压缩，停止迭代
                （2）计算需要删除的 token ，得到需要切分的索引点，删除 history 的前缀消息
                （3）使用 LLM 对前缀消息总结压缩，写入 history.jsonl
                （4）把压缩信息存储起来
                （5）最后返回多个摘要的拼接 str
        """
        if (
            not self.enabled
            or self.context_window_tokens <= 0
            or not history
        ):
            return None

        lock = self._lock(session_key)
        async with lock:
            # budget 计算
            budget = self._input_token_budget
            if budget <= 0:
                return None
            # 压缩目标阈值
            target = max(1024, int(budget * self.consolidation_ratio))

            # 记录每轮压缩生成的摘要字符串
            rollup_parts: list[str] = []
            merged_summary = session_summary

            # 多轮压缩
            for round_num in range(_MAX_CONSOLIDATION_ROUNDS):
                try:
                    # 估算当前 token 数
                    # 每次构建新的消息时，摘要都会合并到 runtime 里的元数据，所以模型是可以看到的
                    estimated, source = self._estimate_prompt(
                        history,
                        chat_id=chat_id,
                        session_summary=merged_summary,
                    )
                except Exception:
                    logger.exception("Consolidator token 估算失败 session={}", session_key)
                    break

                if estimated <= 0:
                    break
                # 低于总预算
                if estimated < budget:
                    logger.debug(
                        "Consolidator idle {}: {}/{} via {}",
                        session_key,
                        estimated,
                        self.context_window_tokens,
                        source,
                    )
                    break

                # 期望删除的 token 数 = 会消耗的 - 阈值（多出来的）
                cut = max(1, estimated - target)
                # 根据 history 列表和 期望删除的 token 得到索引
                boundary = self._pick_boundary(history, cut)
                if boundary is None or boundary <= 0:
                    logger.debug(
                        "Consolidator 无安全切分点 session={} round={}",
                        session_key,
                        round_num,
                    )
                    break

                # 根据索引进行切分
                chunk = history[:boundary]
                logger.info(
                    "Consolidator round {} session={}: est={}/{} via {} chunk={} msgs",
                    round_num,
                    session_key,
                    estimated,
                    self.context_window_tokens,
                    source,
                    len(chunk),
                )
                # 对前缀消息进行压缩，并删除前缀消息
                summary = await self.archive(chunk)
                del history[:boundary]
                self._repair_orphans(history)

                if summary:
                    # 记录压缩的信息
                    rollup_parts.append(summary)
                    merged_summary = "\n\n---\n\n".join(
                        [*(rollup_parts), session_summary] if session_summary else rollup_parts
                    )
                else:
                    merged_summary = session_summary

                if not summary:
                    break

            if not rollup_parts:
                return None
            return "\n\n---\n\n".join(rollup_parts)
