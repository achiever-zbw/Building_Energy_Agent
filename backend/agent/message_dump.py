"""开发调试：把送入模型的消息列表完整打到日志（可选长度上限）。"""

from __future__ import annotations

import json
import os
from typing import Any

from loguru import logger


def log_agent_messages_block(
    tag: str,
    messages: list[dict[str, Any]],
    *,
    iteration: int | None = None,
) -> None:
    """
    将 ``messages`` 以 JSON 缩进格式打印到 INFO。
    环境变量 ``AGENT_LOG_MESSAGES_MAX_CHARS``：大于 0 时对正文截断（默认 0 表示不截断）。
    """
    max_chars_raw = os.environ.get("AGENT_LOG_MESSAGES_MAX_CHARS", "").strip()
    try:
        max_chars = int(max_chars_raw) if max_chars_raw else 0
    except ValueError:
        max_chars = 0
    if max_chars < 0:
        max_chars = 0

    try:
        text = json.dumps(messages, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        text = str(messages)

    note = ""
    if max_chars > 0 and len(text) > max_chars:
        note = f" truncated_was_chars={len(text)} max={max_chars}"
        text = text[:max_chars]

    iter_part = f" iteration={iteration}" if iteration is not None else ""
    # 勿用 logger.info("...{}", text)：JSON 里的 { } 会触发 format 解析异常，日志整段丢失。
    header = f"{tag} msg_count={len(messages)}{iter_part}{note}"
    logger.info(header + "\n" + text)
