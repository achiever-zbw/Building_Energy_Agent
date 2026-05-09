"""Dream 调度：定时周期 + 用户消息明确要更新长期记忆时的即时触发（共享串行锁）。"""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress

from loguru import logger

from typing import TYPE_CHECKING

from backend.agent.dream import Dream

if TYPE_CHECKING:
    from backend.agent.loop import AgentLoop

# --- 开关与周期 -----------------------------------------------------------

def _dream_enabled() -> bool:
    raw = (os.getenv("DREAM_ENABLED") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _chat_trigger_enabled() -> bool:
    """对话里「明确要求更新记忆」时是否立刻排队跑一次 Dream。"""
    raw = (os.getenv("DREAM_CHAT_TRIGGER") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _dream_chat_await_before_response() -> bool:
    """命中关键词时是否在本轮 ``run_turn`` 内 ``await`` Dream（默认 True，避免 HTTP 先返回再后台写完导致误以为未更新）。"""
    raw = (os.getenv("DREAM_CHAT_AWAIT") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def _dream_interval_seconds() -> float:
    try:
        hours = float(os.getenv("DREAM_INTERVAL_HOURS") or "2")
    except ValueError:
        hours = 2.0
    return max(60.0, hours * 3600.0)


# --- 关键词（子串匹配，casefold）-----------------------------------------

_DEFAULT_CHAT_TRIGGERS: tuple[str, ...] = (
    # 明确要写长期记忆文件 / Dream 消化 history 的常见说法
    "记入长期记忆",
    "更新长期记忆",
    "写入长期记忆",
    "保存到长期记忆",
    "同步到记忆",
    "请记住",
    "长期记住",  # 与「请记住」区分：用户常说「长期记住 xxx」
    "请长期记住",
    "一直记住",
    "不要忘记",
    "永远不要忘记",
    "牢记",
    "记住这件事",
    "帮我记下来",
    "更新记忆",
    # 偏好 / 风格 / 人设（USER.md、SOUL.md 常见增量）
    "记住风格",
    "请记住风格",
    "记住语气",
    "请记住语气",
    "记住偏好",
    "请记住偏好",
    "记住人设",
    "记住角色",
    "以后就这样回复",
    "remember this",
    "save to long-term memory",
    "remember permanently",
)


def _effective_chat_triggers() -> tuple[str, ...]:
    raw = (os.getenv("DREAM_CHAT_TRIGGERS") or "").strip()
    if raw:
        return tuple(p.strip() for p in raw.split(",") if p.strip())
    extra = (os.getenv("DREAM_CHAT_TRIGGERS_EXTRA") or "").strip()
    if not extra:
        return _DEFAULT_CHAT_TRIGGERS
    more = tuple(p.strip() for p in extra.split(",") if p.strip())
    return _DEFAULT_CHAT_TRIGGERS + more


def user_message_triggers_immediate_dream(text: str) -> bool:
    if not _chat_trigger_enabled() or not _dream_enabled():
        return False
    s = (text or "").strip()
    if not s:
        return False
    hay = s.casefold()
    for needle in _effective_chat_triggers():
        if needle.casefold() in hay:
            return True
    return False


# --- 串行执行（定时与用户触发共用）---------------------------------------

_lock: asyncio.Lock | None = None


def _serial_lock() -> asyncio.Lock:
    global _lock
    if _lock is None:
        _lock = asyncio.Lock()
    return _lock


async def run_dream_once(agent_loop: AgentLoop, *, reason: str) -> None:
    """同一进程内任意时刻最多一个 Dream.run 在执行。"""
    async with _serial_lock():
        try:
            dream = Dream(
                agent_loop.context.memory,
                agent_loop.provider,
                agent_loop.model,
                context_window_tokens=agent_loop.context_window_tokens,
            )
            worked = await dream.run()
            logger.info("Dream 执行结束 reason={} processed={}", reason, worked)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Dream 执行失败 reason={}", reason)


def _schedule_immediate_dream_fire_and_forget(agent_loop: AgentLoop, user_message: str) -> None:
    """后台排队 Dream（HTTP 先返回）；仅当 ``DREAM_CHAT_AWAIT=0`` 时使用。"""
    if not user_message_triggers_immediate_dream(user_message):
        return
    logger.info("Dream: 对话触发即时整理（关键词命中），后台排队执行")

    async def _job() -> None:
        await run_dream_once(agent_loop, reason="chat_intent")

    try:
        asyncio.create_task(_job(), name="dream_chat_intent")
    except RuntimeError:
        logger.warning("Dream 即时触发失败：当前无运行中的事件循环")


async def run_chat_triggered_dream_if_needed(agent_loop: AgentLoop, user_message: str) -> None:
    """命中关键词则跑一次 Dream；默认在本轮内 await，与定时任务仍共用串行锁。"""
    if not user_message_triggers_immediate_dream(user_message):
        return
    if _dream_chat_await_before_response():
        logger.info("Dream: 对话触发即时整理（关键词命中），本轮请求内等待完成后返回")
        await run_dream_once(agent_loop, reason="chat_intent")
    else:
        _schedule_immediate_dream_fire_and_forget(agent_loop, user_message)


async def dream_scheduler_loop(agent_loop: AgentLoop) -> None:
    """定时：每隔 ``DREAM_INTERVAL_HOURS`` 执行一次。"""
    if not _dream_enabled():
        logger.info("Dream 定时任务未启用（DREAM_ENABLED=0/false）")
        return

    interval_s = _dream_interval_seconds()
    hours = interval_s / 3600.0
    logger.info("Dream 定时任务已启动：约每 {:.2f} 小时执行一次", hours)

    while True:
        try:
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            logger.info("Dream 定时任务已取消")
            raise

        await run_dream_once(agent_loop, reason="schedule")


def spawn_dream_scheduler(agent_loop: AgentLoop) -> asyncio.Task[None]:
    return asyncio.create_task(dream_scheduler_loop(agent_loop), name="dream_scheduler")


async def cancel_task(task: asyncio.Task[None] | None) -> None:
    if task is None or task.done():
        return
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task
