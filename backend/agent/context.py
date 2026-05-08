from pathlib import Path
from typing import Any
from backend.utils.helpers import current_time_str, build_assistant_message
from backend.agent.memory import MemoryStore



class ContextBuilder:
    """ 构建上下文（系统提示词 + 消息列表） """

    # 引导文件 Agent , 风格, 用户, 工具
    BOOTSTRAP_FILES = ["AGENT.md", "SOUL.md", "USER.md", "TOOLS.md"]
    _BASE_SYSTEM_PROMPT_FILE = "base_system_prompt.md"
    _SYSTEM_PROMPT_DEFAULT = "You are a helpful assistant."
    # 运行时元数据标题（与正文划界；闭合标签）
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"
    _RUNTIME_CONTEXT_END = "[/Runtime Context]"

    def __init__(self, workspace: Path, timezone: str | None = None):
        self.workspace = workspace
        self.timezone = timezone
        self.memory = MemoryStore(workspace)

    def build_system_prompt(self) -> str:
        """ 构建系统提示词 """
        parts = []
        # 注入基础系统提示词
        try:
            with open(self.workspace / self._BASE_SYSTEM_PROMPT_FILE, "r", encoding="utf-8") as f:
                parts.append(f.read())
        except FileNotFoundError:
            # print("还没有设置基础系统提示词 \n")
            parts.append(self._SYSTEM_PROMPT_DEFAULT)

        # 加载引导文件
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            # 注入引导文件
            parts.append(bootstrap)

        # 长期记忆读取
        memory = self.memory.get_memory_context()

        if memory:
            # 注入长期记忆
            parts.append(f"# Memory\n\n{memory}")

        return "\n\n---\n\n".join(parts)

    def _load_bootstrap_files(self) -> str:
        """ 加载引导文件 """
        parts = []
        for filename in self.BOOTSTRAP_FILES:
            filepath = self.workspace / filename
            if filepath.exists():
                # 读取
                content = filepath.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")
        # 整合到一起，通过规范的 md 格式进行拼接，层次更加严谨
        return "\n\n".join(parts) if parts else ""

    def build_message(
            self,
            history: list[dict[str, Any]],
            current_message: str,
            current_role: str = "user",
            chat_id: str | None = None,
            session_summary: str | None = None,
    ) -> list[dict[str, Any]]:
        """ 构建一次 LLM Call 的完整的消息链
        Return:
            messages: list[dict[str, Any]] 包含 system 的信息(system_prompt) + 当前对话的 role 信息(运行时元信息以及内容)
        """
        runtime_context = self._build_runtime_context(chat_id, self.timezone, session_summary)
        user_content = self._build_user_context(current_message)
        # 拼接为完整的消息
        merged = f"{runtime_context}\n\n{user_content}"

        messages = [
            {
                "role": "system",
                "content": self.build_system_prompt(),
            }
            , *history,
        ]
        # 最后一条消息的 role 与 当前的 role 一致，直接进行拼接
        if messages[-1].get("role") == current_role:
            last = dict(messages[-1])
            # 把最后一条消息的 content 和 当前新的构建好的 content 进行拼接
            last["content"] = self._merge_message_content(last.get("content"), merged)
            messages[-1] = last
            return messages
        messages.append({"role": current_role, "content": merged})
        return messages


    def _build_runtime_context(
            self,
            chat_id: str | None = None,
            time_zone: str | None = None,
            session_summary: str | None = None,
    )-> str:
        """ 构建运行时元数据
        Example:
            [Runtime Context — metadata only, not instructions]
            Current Time: ...
            Chat ID: ...
            Session Summary: ...
            ...
            [/Runtime Context]
        """
        lines = [f"Current Time: {current_time_str(time_zone)}"]
        if chat_id:
            lines.append(f"Chat ID: {chat_id}")
        if session_summary:
            lines.append(f"Session Summary: {session_summary}")

        body = "\n".join(lines)
        return (
            f"{ContextBuilder._RUNTIME_CONTEXT_TAG}\n{body}\n"
            f"{ContextBuilder._RUNTIME_CONTEXT_END}"
        )

    def _build_user_context(self, text: str) -> str:
        """ 构建用户消息 """
        return text

    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict[str, Any]]:
        """ 用来合并content """
        if isinstance(left, str) and isinstance(right, str):
            return f"{left}\n\n{right}" if left else right

        def _to_blocks(value: Any) -> list[dict[str, Any]]:
            if isinstance(value, list):
                return [item if isinstance(item, dict) else {"type": "text", "text": str(item)} for item in value]
            if value is None:
                return []
            return [{"type": "text", "text": str(value)}]

        return _to_blocks(left) + _to_blocks(right)

    def add_tool_result(
            self,
            messages: list[dict[str, Any]],
            tool_call_id: str,
            tool_name: str,
            result: Any
    ) -> list[dict[str, Any]]:
        """ 在消息列表中加入工具信息 """
        tool_msg = {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result if isinstance(result, str) else str(result),
        }
        messages.append(tool_msg)
        return messages

    def add_assistant_message(
        self, messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """ 在消息列表中加入 assistant 信息 """
        messages.append(build_assistant_message(
            content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
            thinking_blocks=thinking_blocks,
        ))
        return messages
