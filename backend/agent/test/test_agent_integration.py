"""
集成行为测试（无需真实 API Key）：

1. 长短期上下文 — SYSTEM 含 MEMORY.md；history 进入 messages。
2. 压缩 — Consolidator 超预算时归档前缀并缩短 history，写入 history.jsonl。
3. 多工具 — AgentLoop 在一轮内执行多个 tool_calls，或多轮串联执行。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from backend.agent.context import ContextBuilder
from backend.agent.consolidator import Consolidator
from backend.agent.loop import AgentLoop
from backend.agent.schemas import LLMResponse, ToolCallRequest
from backend.agent.tools.base import Tool
from backend.agent.tools.registry import ToolRegistry
from backend.agent.test.mock_provider import ScriptedProvider


class EchoTool(Tool):
    @property
    def name(self) -> str:
        return "echo_tool"

    @property
    def description(self) -> str:
        return "回显 text"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(self, **kwargs):
        return kwargs.get("text", "")


def test_long_term_memory_in_system_prompt(agent_workspace: Path) -> None:
    """长期：MEMORY.md 注入 system；短期不在 system。"""
    ws = agent_workspace
    marker = "LT_MEMORY_UNIQUE_MARKER_90123"
    (ws / "memory" / "MEMORY.md").write_text(f"# Notes\n\n{marker}\n", encoding="utf-8")

    ctx = ContextBuilder(ws)
    system = ctx.build_system_prompt()
    assert marker in system
    assert "# Memory" in system


def test_short_term_history_passed_to_messages(agent_workspace: Path) -> None:
    """短期：多轮 user/assistant 保留在 initial_messages 中（不含 system）。"""
    ws = agent_workspace
    ctx = ContextBuilder(ws)
    hist = [
        {"role": "user", "content": "第一轮问题"},
        {"role": "assistant", "content": "第一轮答"},
    ]
    messages = ctx.build_message(hist, "第二轮追问", chat_id="chat-1")
    assert messages[0]["role"] == "system"
    bodies = json.dumps(messages, ensure_ascii=False)
    assert "第一轮问题" in bodies and "第一轮答" in bodies
    assert "第二轮追问" in bodies
    assert "chat-1" in bodies


def test_consolidator_archives_prefix_and_shortens_history(
    agent_workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """超 token 预算时：归档前缀、history 变短、history.jsonl 出现 Consolidated 记录。"""
    ws = agent_workspace
    ctx = ContextBuilder(ws)
    registry = ToolRegistry()

    archive_summary = "【压缩摘要】用户曾询问建筑 A 与 B 的能耗。"
    provider = ScriptedProvider(
        [LLMResponse(content=archive_summary, finish_reason="stop")]
    )

    cons = Consolidator(
        store=ctx.memory,
        provider=provider,
        model="mock-model",
        context_window_tokens=12_000,
        context_builder=ctx,
        tools=registry,
        consolidation_ratio=0.5,
        max_completion_tokens=2048,
    )

    calls = {"n": 0}

    def fake_estimate(self, history, *, chat_id, session_summary):  # noqa: ARG001
        calls["n"] += 1
        if calls["n"] == 1:
            return 200_000, "forced-overflow"
        return 800, "below-budget"

    monkeypatch.setattr(Consolidator, "_estimate_prompt", fake_estimate)

    hist = [
        {"role": "user", "content": "较早的对话起点"},
        {"role": "assistant", "content": "较早回复"},
        {"role": "user", "content": "较近的问题"},
        {"role": "assistant", "content": "较近回复"},
    ]
    orig_len = len(hist)

    async def _run() -> str | None:
        return await cons.maybe_consolidate_history(
            hist,
            chat_id="t1",
            session_summary=None,
            session_key="t1",
        )

    rollup = asyncio.run(_run())

    assert rollup is not None
    assert archive_summary in rollup
    assert len(hist) < orig_len

    raw = ctx.memory.history_file.read_text(encoding="utf-8")
    assert "[Consolidated" in raw or "[RAW consolidation]" in raw


def test_consolidator_disabled_no_op(agent_workspace: Path) -> None:
    ws = agent_workspace
    ctx = ContextBuilder(ws)
    registry = ToolRegistry()
    provider = ScriptedProvider([])

    cons = Consolidator(
        store=ctx.memory,
        provider=provider,
        model="m",
        context_window_tokens=128_000,
        context_builder=ctx,
        tools=registry,
        enabled=False,
    )

    async def _run() -> None:
        hist = [{"role": "user", "content": "x"}]
        before = list(hist)
        r = await cons.maybe_consolidate_history(
            hist, chat_id=None, session_summary=None, session_key="k"
        )
        assert r is None
        assert hist == before

    asyncio.run(_run())


def test_agent_loop_multiple_tool_calls_single_iteration(agent_workspace: Path) -> None:
    """同一轮模型返回两个 tool_calls，应全部执行并得到最终自然语言答复。"""
    ws = agent_workspace
    reg = ToolRegistry()
    reg.register(EchoTool())

    responses = [
        LLMResponse(
            content="并行调用两个 echo",
            finish_reason="tool_calls",
            tool_calls=[
                ToolCallRequest(id="c1", name="echo_tool", arguments={"text": "task-alpha"}),
                ToolCallRequest(id="c2", name="echo_tool", arguments={"text": "task-beta"}),
            ],
        ),
        LLMResponse(content="已收到 task-alpha 与 task-beta，全部完成。", finish_reason="stop"),
    ]
    provider = ScriptedProvider(responses)

    loop = AgentLoop(
        provider=provider,
        workspace=ws,
        tools=reg,
        enable_consolidation=False,
        max_iterations=8,
    )

    async def _run():
        return await loop.run_turn(
            "请并行执行两个回显",
            history=[],
            chat_id="multi",
            consolidate=False,
            persist_history=False,
        )

    result = asyncio.run(_run())

    assert result.stop_reason == "completed"
    assert result.tools_used.count("echo_tool") == 2
    assert result.final_content and "task-alpha" in result.final_content


def test_agent_loop_chained_tool_rounds(agent_workspace: Path) -> None:
    """两轮工具调用：第一轮 echo step1，第二轮 echo step2，最后汇总。"""
    ws = agent_workspace
    reg = ToolRegistry()
    reg.register(EchoTool())

    responses = [
        LLMResponse(
            content="先做第一步",
            finish_reason="tool_calls",
            tool_calls=[
                ToolCallRequest(id="t1", name="echo_tool", arguments={"text": "step-one"}),
            ],
        ),
        LLMResponse(
            content="再做第二步",
            finish_reason="tool_calls",
            tool_calls=[
                ToolCallRequest(id="t2", name="echo_tool", arguments={"text": "step-two"}),
            ],
        ),
        LLMResponse(content="两步工具均已执行：step-one → step-two。", finish_reason="stop"),
    ]
    provider = ScriptedProvider(responses)

    loop = AgentLoop(
        provider=provider,
        workspace=ws,
        tools=reg,
        enable_consolidation=False,
        max_iterations=8,
    )

    async def _run():
        return await loop.run_turn(
            "串联两步 echo",
            history=[],
            chat_id="chain",
            consolidate=False,
            persist_history=False,
        )

    result = asyncio.run(_run())

    assert result.stop_reason == "completed"
    assert result.tools_used == ["echo_tool", "echo_tool"]
    assert "step-one" in (result.final_content or "")
    assert "step-two" in (result.final_content or "")


def test_persist_history_appends_jsonl(agent_workspace: Path) -> None:
    """persist_history=True 时向 history.jsonl 追加用户句与助手最终句。"""
    ws = agent_workspace
    reg = ToolRegistry()
    reg.register(EchoTool())

    provider = ScriptedProvider(
        [
            LLMResponse(
                content="",
                finish_reason="tool_calls",
                tool_calls=[
                    ToolCallRequest(id="x", name="echo_tool", arguments={"text": "hi"}),
                ],
            ),
            LLMResponse(content="助手最终答复 persist-test", finish_reason="stop"),
        ]
    )

    loop = AgentLoop(
        provider=provider,
        workspace=ws,
        tools=reg,
        enable_consolidation=False,
    )

    async def _run():
        await loop.run_turn(
            "用户句 persist-user",
            history=[],
            chat_id="persist",
            consolidate=False,
            persist_history=True,
        )

    asyncio.run(_run())

    hf = ws / "memory" / "history.jsonl"
    lines = [
        json.loads(line)
        for line in hf.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    contents = " ".join(str(rec.get("content", "")) for rec in lines)
    assert "用户句 persist-user" in contents
    assert "persist-test" in contents