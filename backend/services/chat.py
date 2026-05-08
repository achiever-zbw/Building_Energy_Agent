from dataclasses import dataclass
from typing import Any

@dataclass
class ChatSession:
    """统一聊天会话，单通道 MCP tools。"""

    mcp_client: Any

    async def ask(self, message: str) -> str:
        """处理一轮对话，统一使用 MCPClient.messages 维护上下文。"""
        if not message or not message.strip():
            return "请输入问题。"
        return await self.mcp_client.process_query(message)

    def clear(self) -> None:
        """清空会话上下文。"""
        self.mcp_client.reset_messages()