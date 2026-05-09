"""长期记忆 Dream：两阶段处理 history.jsonl 未读条目，再经工具增量更新 MEMORY/SOUL/USER。

对齐 nanobot Dream 流程；不包含 Git、skills、MEMORY 行龄标注。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from backend.agent.runner import AgentRunner
from backend.agent.schemas import AgentRunSpec
from backend.agent.tools.registry import ToolRegistry
from backend.agent.tools.workspace_tools import build_dream_workspace_tools
from backend.utils.helpers import truncate_text

if TYPE_CHECKING:
    from backend.agent.memory import MemoryStore
    from backend.agent.providers.base import LLMProvider

# 三个记忆文件以及角色
_MEMORY_FILE_MAX_CHARS = 32_000
_SOUL_FILE_MAX_CHARS = 16_000
_USER_FILE_MAX_CHARS = 16_000
_HISTORY_ENTRY_PREVIEW_MAX_CHARS = 4_000

_MAX_CONSOLIDATION_PHASE1_TOKENS = 8192


DREAM_PHASE1_SYSTEM = """你是「长期记忆整理」分析助手（Phase 1）。

输入包含：
- 一段按时间排序的对话/历史摘录（来自 history.jsonl）；
- 当前的 MEMORY.md、SOUL.md、USER.md 摘要（可能截断）。

任务：
1. 从中抽取**值得写入长期记忆**的稳定事实：用户偏好、重复出现的实体、约定、未完成任务、项目结论等。
2. 区分：适合 MEMORY.md 的事实笔记、适合 USER.md 的用户偏好、适合 SOUL.md 的助手行为原则（若对话未涉及则可不写）。
3. 指出与现有三文件内容的**重复或冲突**，建议合并而非堆砌。
4. 输出清晰的可执行「整理说明」，供下一阶段使用工具逐文件编辑；不要编造对话中不存在的内容。

使用与用户相同的语言输出。"""

DREAM_PHASE2_SYSTEM = """你是「长期记忆整理」执行助手（Phase 2）。

你已收到 Phase 1 的分析结论，以及当前 MEMORY/SOUL/USER 的摘要预览。
你必须使用工具：
- `read_workspace_file`：读取将要修改文件的完整内容；
- `edit_workspace_file`：仅在 **memory/MEMORY.md、SOUL.md、USER.md** 上做**小范围**替换（old_string 必须与文件原文完全一致）。

规则：
- 禁止改写 workspace 中其它路径；禁止臆造文件内容。
- 每次修改前先 read，再 edit；old_string 要在文件中唯一匹配（除非有意 replace_all）。
- 保持 Markdown 结构可读；不要无谓清空整文件。
- 若分析结论认为无需改动任何文件，可直接简短回复说明无需工具并完成。

使用与用户相同的语言思考；工具调用与文件内容可使用原有语言。"""


class Dream:
    """从未处理的 history.jsonl 批次触发：Phase1 分析 → Phase2 ReAct 写文件。"""

    def __init__(
        self,
        store: MemoryStore,
        provider: LLMProvider,
        model: str,
        *,
        max_batch_size: int = 20,
        max_iterations: int = 10,
        max_tool_result_chars: int = 16_000,
        context_window_tokens: int = 128_000,
    ) -> None:
        self.store = store
        self.provider = provider
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_iterations = max_iterations
        self.max_tool_result_chars = max_tool_result_chars
        self.context_window_tokens = context_window_tokens
        self._runner = AgentRunner(provider)

    def set_provider(self, provider: LLMProvider, model: str) -> None:
        self.provider = provider
        self.model = model
        self._runner.provider = provider

    def _build_tools(self) -> ToolRegistry:
        tools = ToolRegistry()
        read_t, edit_t = build_dream_workspace_tools(self.store.workspace)
        tools.register(read_t)
        tools.register(edit_t)
        return tools

    async def run(self) -> bool:
        """处理 dream 游标之后的 history 条目；有批次则返回 True。"""
        # 获取上一次 dream 长期记忆处理到的位置
        last_cursor = self.store.get_last_dream_cursor()
        # 读取新的历史消息（最新游标位置到最新的历史消息）
        entries = self.store.read_unprocessed_history(since_cursor=last_cursor)
        if not entries:
            return False

        # 批量处理，默认最多 20 条
        batch = entries[: self.max_batch_size]
        logger.info(
            "Dream: processing {} entries (cursor {}→{}), batch={}",
            len(entries),
            last_cursor,
            batch[-1]["cursor"],
            len(batch),
        )

        # 构建历史消息文本
        history_text = "\n".join(
            f"[{e['timestamp']}] "
            f"{truncate_text(str(e.get('content', '')), _HISTORY_ENTRY_PREVIEW_MAX_CHARS)}"
            for e in batch
        )

        # 读取当前的记忆
        current_date = datetime.now().strftime("%Y-%m-%d")
        raw_memory = self.store.read_memory() or "(empty)"
        current_memory = truncate_text(raw_memory, _MEMORY_FILE_MAX_CHARS)
        current_soul = truncate_text(self.store.read_soul() or "(empty)", _SOUL_FILE_MAX_CHARS)
        current_user = truncate_text(self.store.read_user() or "(empty)", _USER_FILE_MAX_CHARS)

        # 构建完整的记忆文本
        file_context = (
            f"## Current Date\n{current_date}\n\n"
            f"## Current MEMORY.md ({len(current_memory)} chars)\n{current_memory}\n\n"
            f"## Current SOUL.md ({len(current_soul)} chars)\n{current_soul}\n\n"
            f"## Current USER.md ({len(current_user)} chars)\n{current_user}"
        )

        phase1_user = f"## Conversation History\n{history_text}\n\n{file_context}"

        try:
            # phase 1 分析阶段，把总结历史信息的需求和完整上下文给到 LLM 
            phase1_response = await self.provider.chat_with_retry(
                model=self.model,
                messages=[
                    {"role": "system", "content": DREAM_PHASE1_SYSTEM},
                    {"role": "user", "content": phase1_user},
                ],
                tools=None,
                tool_choice=None,
                max_tokens=_MAX_CONSOLIDATION_PHASE1_TOKENS,
                temperature=0.2,
            )
            if phase1_response.finish_reason == "error":
                raise RuntimeError(phase1_response.content or "LLM error")
            analysis = (phase1_response.content or "").strip()
            logger.debug("Dream Phase 1 analysis ({} chars)", len(analysis))
        except Exception:
            logger.exception("Dream Phase 1 failed")
            return False

        # phase 2 阶段，拿到分析结果给 LLM ，来执行更新上下文
        phase2_user = f"## Analysis Result\n{analysis}\n\n{file_context}"
        tools = self._build_tools()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": DREAM_PHASE2_SYSTEM},
            {"role": "user", "content": phase2_user},
        ]

        try:
            # 此时传递完整上下文，以及两个记忆更新工具
            spec = AgentRunSpec(
                initial_messages=messages,
                tools=tools,
                model=self.model,
                max_iterations=self.max_iterations,
                max_tool_result_chars=self.max_tool_result_chars,
                temperature=0.2,
                workspace=self.store.workspace,
                fail_on_tool_error=False,
                context_window_tokens=self.context_window_tokens,
            )
            result = await self._runner.run(spec)
            logger.debug(
                "Dream Phase 2: stop_reason={}, tool_events={}",
                result.stop_reason,
                len(result.tool_events or []),
            )
        except Exception:
            logger.exception("Dream Phase 2 failed")
            result = None

        # 日志记录变更
        changelog: list[str] = []
        if result and result.tool_events:
            for event in result.tool_events:
                if event.get("status") == "ok":
                    changelog.append(f"{event.get('name')}: {event.get('detail', '')}")
        
        # 更新最新游标位置
        new_cursor = int(batch[-1]["cursor"])
        self.store.set_last_dream_cursor(new_cursor)
        self.store.compact_history()

        if result and result.stop_reason == "completed":
            logger.info(
                "Dream done: {} tool ok event(s), cursor → {}",
                len(changelog),
                new_cursor,
            )
        else:
            reason = getattr(result, "stop_reason", None) if result else "exception"
            logger.warning(
                "Dream incomplete ({}): cursor advanced to {}",
                reason,
                new_cursor,
            )

        return True
