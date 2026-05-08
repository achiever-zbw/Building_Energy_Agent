from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """前端发来的对话请求。"""

    message: str = Field(..., min_length=1, description="用户本轮输入")
    session_id: str = Field(
        default="default",
        description="会话 ID；同一 ID 多轮对话共享上下文",
    )


class ChatResponse(BaseModel):
    """返回给前端的结构化回复。"""

    reply: str = ""
    session_id: str = "default"
    stop_reason: str = "completed"
    usage: dict[str, int] = Field(default_factory=dict)
