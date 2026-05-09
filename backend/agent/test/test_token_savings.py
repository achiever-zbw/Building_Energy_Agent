"""
Token 节省与降低率（尽量少 mock）：

- **短期压缩（真实场景）**：用与线上一致的 ``estimate_prompt_tokens_chain`` + 真实 ``build_default_tool_registry()``
  增大 history，直到估算 prompt ≥ Consolidator 预算，触发真实 ``maybe_consolidate_history``；归档摘要走 **真实 API**
  （环境变量 ``OPENAI_API_KEY`` 或 ``DEEPSEEK_API_KEY``；``pytest`` 经 ``conftest`` 加载仓库 ``.env``），标记 ``@pytest.mark.integration``。

- **长期 MEMORY**：仅用 tiktoken 估算「system 注入一次」vs「每轮 user 重复粘贴」的 prompt 体积差异，不调用模型。
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest

from backend.agent.consolidator import Consolidator
from backend.agent.context import ContextBuilder
from backend.agent.providers.openai_provider import OpenAIProvider
from backend.agent.tools.registry import ToolRegistry
from backend.agent.tools.tools import build_default_tool_registry
from backend.utils.helpers import (
    estimate_prompt_tokens_chain,
    openai_compatible_model_from_env,
)


def _resolve_live_llm_credentials() -> tuple[str, str | None, str]:
    """与线上一致：优先 OPENAI_*，否则 DeepSeek 兼容网关。"""
    oai = (os.getenv("OPENAI_API_KEY") or "").strip()
    if oai:
        base = (os.getenv("OPENAI_API_BASE") or "").strip() or None
        model = (openai_compatible_model_from_env() or "").strip()
        if not model:
            # 常见坑：网关指向 DeepSeek 但未配置模型名时，勿默认 gpt-4o-mini
            if base and "deepseek" in base.lower():
                model = "deepseek-chat"
            else:
                model = "gpt-4o-mini"
        return oai, base, model

    dsk = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if dsk:
        base = (os.getenv("OPENAI_API_BASE") or os.getenv("DEEPSEEK_API_BASE") or "").strip()
        if not base:
            base = "https://api.deepseek.com"
        model = (openai_compatible_model_from_env() or "").strip() or "deepseek-chat"
        return dsk, base, model

    return "", None, "gpt-4o-mini"


def _tiktoken_provider(model: str | None = None) -> OpenAIProvider:
    """仅参与 tiktoken 分支估算；若不调用 ``chat`` 则不会走 HTTP。"""
    api_key, api_base, default_m = _resolve_live_llm_credentials()
    if not api_key:
        api_key = "sk-placeholder-no-http-in-this-test"
    base = api_base if api_base is not None else "https://api.openai.com/v1"
    return OpenAIProvider(
        api_key=api_key,
        api_base=base,
        default_model=model or default_m,
    )


def _estimate_turn_prompt(
    ctx: ContextBuilder,
    provider: Any,
    model: str,
    registry: ToolRegistry,
    history: list[dict[str, Any]],
    *,
    chat_id: str,
    session_summary: str | None,
) -> int:
    probe = ctx.build_message(
        list(history),
        "[token-probe]",
        current_role="user",
        chat_id=chat_id,
        session_summary=session_summary,
    )
    tokens, _ = estimate_prompt_tokens_chain(
        provider, model, probe, registry.get_definitions()
    )
    return int(tokens)


def _consolidator_input_budget(
    *, context_window_tokens: int, max_completion_tokens: int, safety_buffer: int = 1024
) -> int:
    return context_window_tokens - max(1, max_completion_tokens) - safety_buffer


def _grow_history_until_at_least_budget(
    ctx: ContextBuilder,
    provider: Any,
    model: str,
    registry: ToolRegistry,
    *,
    chat_id: str,
    budget: int,
    session_summary: str | None = None,
    max_pairs: int = 120,
) -> list[dict[str, Any]]:
    """递增 user/assistant 对，直到「system + tools + history + probe」估算 tokens ≥ budget。"""
    hist: list[dict[str, Any]] = []
    filler = "能耗分项 electricity/chilledwater 累计；" * 60
    for _ in range(max_pairs):
        t = _estimate_turn_prompt(
            ctx,
            provider,
            model,
            registry,
            hist,
            chat_id=chat_id,
            session_summary=session_summary,
        )
        if t >= budget:
            return hist
        idx = len(hist) // 2
        hist.append({"role": "user", "content": f"轮次{idx} 查询与上下文。\n{filler}"})
        hist.append({"role": "assistant", "content": f"轮次{idx} 答复与表格摘要。\n{filler}"})
    pytest.fail(
        f"在 {max_pairs} 轮内未能使估算 tokens≥budget={budget}，请加大 filler 或提高工具 schema 权重后再试"
    )


@pytest.mark.integration
def test_consolidation_real_trigger_and_live_archive_reduces_tokens(
    agent_workspace: Path,
) -> None:
    """真实压缩：tiktoken 自然超预算 + 真实 LLM 归档；断言 history 变短且估算 prompt 下降。"""
    api_key, api_base, model_default = _resolve_live_llm_credentials()
    if not api_key:
        pytest.skip(
            "需要 OPENAI_API_KEY 或 DEEPSEEK_API_KEY（可在仓库根目录 .env 中配置；已由 conftest 加载）"
        )

    ctx = ContextBuilder(agent_workspace)
    registry = build_default_tool_registry()
    provider = OpenAIProvider(
        api_key=api_key,
        api_base=api_base,
        default_model=model_default,
    )
    model = provider.default_model

    # 缩小名义上下文窗口，使可用输入预算较低，便于用有限轮对话触发压缩（仍含真实工具定义开销）
    context_window_tokens = 14_000
    max_completion_tokens = 2048
    budget = _consolidator_input_budget(
        context_window_tokens=context_window_tokens,
        max_completion_tokens=max_completion_tokens,
    )
    assert budget > 0

    cons = Consolidator(
        store=ctx.memory,
        provider=provider,
        model=model,
        context_window_tokens=context_window_tokens,
        context_builder=ctx,
        tools=registry,
        consolidation_ratio=0.5,
        max_completion_tokens=max_completion_tokens,
    )

    chat_id = "integration-consolidation"
    hist = _grow_history_until_at_least_budget(
        ctx,
        provider,
        model,
        registry,
        chat_id=chat_id,
        budget=budget,
    )

    before = _estimate_turn_prompt(
        ctx, provider, model, registry, hist, chat_id=chat_id, session_summary=None
    )
    assert before >= budget, "应先满足 Consolidator 触发条件（估算 ≥ 输入预算）"
    orig_len = len(hist)

    rollup = asyncio.run(
        cons.maybe_consolidate_history(
            hist,
            chat_id=chat_id,
            session_summary=None,
            session_key=chat_id,
        )
    )

    assert rollup, "归档应返回 Session Summary 片段（真实模型成功时非空）"
    assert len(hist) < orig_len, "压缩应删除已归档的前缀消息"

    after = _estimate_turn_prompt(
        ctx,
        provider,
        model,
        registry,
        hist,
        chat_id=chat_id,
        session_summary=rollup,
    )

    assert after < before, (
        f"压缩后估算 prompt 应变小：before={before}, after={after}, "
        f"rollup_len={len(rollup)}"
    )
    reduction_rate = (before - after) / max(before, 1)
    assert reduction_rate > 0.05, (
        f"降低率应明显为正（阈值 5% 可调）：{reduction_rate:.2%}"
    )


def test_long_term_memory_vs_repeated_user_paste_token_gap() -> None:
    """长期：同等信息量只在 system（MEMORY）出现一次 vs 每轮 user 重复粘贴 — 仅 tiktoken，不调 API。"""
    repo_root = Path(__file__).resolve().parents[3]
    provider = _tiktoken_provider()
    model = provider.default_model
    registry = ToolRegistry()

    with TemporaryDirectory(dir=repo_root) as td:
        ws = Path(td) / "workspace"
        ws.mkdir()
        (ws / "memory").mkdir()
        memory_blob = "偏好与备忘：" + ("【长期记忆条目】" * 80)
        (ws / "memory" / "MEMORY.md").write_text(memory_blob, encoding="utf-8")

        ctx = ContextBuilder(ws)

        rounds = 8
        history_lt: list[dict[str, Any]] = []
        for i in range(rounds):
            history_lt.append({"role": "user", "content": f"第{i}轮简短问题"})
            history_lt.append({"role": "assistant", "content": f"第{i}轮简短答"})

        tokens_lt = _estimate_turn_prompt(
            ctx,
            provider,
            model,
            registry,
            history_lt,
            chat_id="lt",
            session_summary=None,
        )

        history_naive: list[dict[str, Any]] = []
        for i in range(rounds):
            history_naive.append(
                {"role": "user", "content": f"{memory_blob}\n\n第{i}轮问题"}
            )
            history_naive.append({"role": "assistant", "content": f"第{i}轮答"})

        ws2 = Path(td) / "workspace2"
        ws2.mkdir()
        (ws2 / "memory").mkdir()
        ctx_naive = ContextBuilder(ws2)
        tokens_naive = _estimate_turn_prompt(
            ctx_naive,
            provider,
            model,
            registry,
            history_naive,
            chat_id="naive",
            session_summary=None,
        )

        assert tokens_naive > tokens_lt * 2, (
            "重复把长期正文粘在每条 user 上应远大于只在 system/Memory 注入一次；"
            f"tokens_lt={tokens_lt}, tokens_naive={tokens_naive}"
        )
        ratio_saved_vs_naive = (tokens_naive - tokens_lt) / max(tokens_naive, 1)
        assert ratio_saved_vs_naive > 0.4
