"""
Agent 对话路由要点（对齐 AgentLoop.run_turn）：

1. AgentLoop 在应用启动时创建一次，挂在 ``request.app.state.agent_loop``（见 ``main.py``）。
2. 多轮对话的 ``history`` 不能包含 ``role == \"system\"`` 的消息：
   ``ContextBuilder.build_message`` 会自己拼 system，再由 Runner 在单次回合内追加工具轨迹。
3. 每一轮结束后，用 ``result.messages`` 去掉 system 后的列表覆盖会话缓存，
   下一轮把整个列表当作 ``history`` 传入 ``run_turn``。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request
from backend.schemas.chat import ChatRequest, ChatResponse

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


def _history_without_system(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Runner 返回的 messages 含 system；存会话时需去掉，避免下一轮重复 system。"""
    return [m for m in messages if m.get("role") != "system"]


@router.post("", summary="Agent 对话（ReAct）", response_model=ChatResponse)
async def chat_agent(request: Request, body: ChatRequest = Body(...)):
    loop = getattr(request.app.state, "agent_loop", None)
    if loop is None:
        raise HTTPException(
            status_code=503,
            detail="Agent 未初始化：请配置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY 后重启服务",
        )

    if not hasattr(request.app.state, "chat_sessions"):
        request.app.state.chat_sessions = {}
    sessions: dict[str, list[dict[str, Any]]] = request.app.state.chat_sessions

    sid = (body.session_id or "default").strip() or "default"
    history = list(sessions.get(sid, []))

    try:
        result = await loop.run_turn(
            body.message.strip(),
            history=history,
            chat_id=sid,
            persist_history=True,
            consolidate=True,
            session_key=sid,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Agent 执行异常: {e!s}") from e

    sessions[sid] = _history_without_system(result.messages)

    reply = (result.final_content or "").strip()
    if not reply and result.error:
        reply = result.error.strip()

    return ChatResponse(
        reply=reply,
        session_id=sid,
        stop_reason=result.stop_reason,
        usage=dict(result.usage or {}),
    )
