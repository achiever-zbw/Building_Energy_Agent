from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from backend.agent.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from backend.agent.hook import AgentHook

from pathlib import Path


_DEFAULT_ERROR_MESSAGE = "抱歉，智能分析服务暂时无法使用，请稍后再试"
_MAX_ITERATIONS_MESSAGE = "本次问题需要的工具调用轮次过多，已停止继续执行。请补充更明确的问题后重试。"


@dataclass(slots=True)
class AgentRunSpec:
    """ 一次 Agent 运行的规范约束 """
    # 传递给模型的‍初始消息
    initial_messages: list[dict[str, Any]]

    # 工具注册表，负责提供工具 schema, 参数，工具执行
    tools: ToolRegistry

    # 模型名；为 None 时由 Runner 使用 provider.get_default_model()
    model: str | None = None

    # 最大 ReAct 迭代次数
    max_iterations: int = 15

    # 单个工具结果允许回填给模型的最大字符数量，防止上下文爆炸
    max_tool_result_chars: int = 32_000

    # 模型温度；为 None 时用 Provider 默认（如 GenerationSettings）
    temperature: float | None = None

    # 工作区
    workspace: Path | None = None

    # 指定的最大输出 Token
    max_tokens: int | None = None

    #
    context_block_limit: int | None = None

    # 部分模型支持的推理力度
    reasoning_effort: str | None = None

    # 生命周期 Hook，用于日志、工具轨迹等
    hook: AgentHook | None = None

    # Agent 执行失败的兜底错误信息
    error_message: str = _DEFAULT_ERROR_MESSAGE

    # 迭代次数达到最大时返回给用户
    max_iterations_message: str = _MAX_ITERATIONS_MESSAGE

    # 上下文窗口的token
    context_window_tokens: int | None = None

    # 工具失败是否直接终止
    # False 表示工具错误回填给模型，尝试修正
    fail_on_tool_error: bool = False

    # 模型调用超时时间
    llm_timeout_s: float | None = None

    # 可选：runner 各阶段回调（如 UI 恢复）
    checkpoint_callback: Callable[[dict[str, Any]], Awaitable[None]] | None = None

    # 是否允许并发执行标记为 concurrency_safe 的工具批次
    concurrent_tools: bool = False
@dataclass(slots=True)
class AgentRunResult:
    """ Agent 运行结果表单 """
    # 最终返回给用户的信息
    final_content: str | None

    # 本次运行 agent 产生的消息轨迹
    messages: list[dict[str, Any]]

    # 本次运行调用过的工具名（按出现顺序）
    tools_used: list[str] = field(default_factory=list)

    # 累计消耗的 Token（键名随 Provider，常见 prompt_tokens / completion_tokens）
    usage: dict[str, int] = field(default_factory=dict)

    # 停止原因：completed | max_iterations | tool_error | error | empty_final_response 等
    stop_reason: str = "completed"

    # 错误信息
    error: str | None = None

    # 每条工具的简要事件（name / status / detail），便于日志或 UI；不需要可忽略
    tool_events: list[dict[str, str]] = field(default_factory=list)

    # 若以后做 injection 队列，可为 True；当前 MVP 可始终 False
    had_injections: bool = False


@dataclass
class ToolCallRequest:
    """ 模型请求调用工具的请求信息体 """
    id: str
    name: str
    arguments: dict[str, Any]
    extra_content: dict[str, Any] | None = None
    provider_specific_fields: dict[str, Any] | None = None
    function_provider_specific_fields: dict[str, Any] | None = None

    # 将 Tool Call 转化为 OpenAI 的接受格式
    def to_openai_tool_call(self) -> dict[str, Any] :
        # 一次 tool call 包含：id , type , function(name , args)
        tool_call = {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }
        # 额外信息的补充
        if self.extra_content:
            tool_call["extra_content"] = self.extra_content
        if self.provider_specific_fields:
            tool_call["provider_specific_fields"] = self.provider_specific_fields
        if self.function_provider_specific_fields:
            tool_call["function_provider_specific_fields"] = self.function_provider_specific_fields

        return tool_call


@dataclass
class LLMResponse:
    """Response from an LLM provider."""
    # LLM 返回的内容
    content: str | None

    # 请求执行的工具列表
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    # 结束原因，stop(自然结束) length(达到 max_tokens) tool_calls(要求执行工具) error(发生错误)
    finish_reason: str = "stop"

    # token 使用量
    usage: dict[str, int] = field(default_factory=dict)

    # 建议的重试等待时间
    retry_after: float | None = None  # Provider supplied retry wait in seconds.

    # 推理内容
    reasoning_content: str | None = None  # Kimi, DeepSeek-R1, MiMo etc.

    # 思考块
    thinking_blocks: list[dict] | None = None  # Anthropic extended thinking

    # Structured error metadata used by retry policy when finish_reason == "error".
    # 错误 HTTP 状态码
    error_status_code: int | None = None

    # 错误类型 timeout connect
    error_kind: str | None = None  # e.g. "timeout", "connection"

    # 模型商提供的错误类型
    error_type: str | None = None  # Provider/type semantic, e.g. insufficient_quota.
    error_code: str | None = None  # Provider/code semantic, e.g. rate_limit_exceeded.
    error_retry_after_s: float | None = None
    error_should_retry: bool | None = None

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0

    @property
    def should_execute_tools(self) -> bool:
        """Tools execute only when has_tool_calls AND finish_reason is ``tool_calls`` / ``stop``.
        Blocks gateway-injected calls under ``refusal`` / ``content_filter`` / ``error`` (#3220)."""
        if not self.has_tool_calls:
            return False
        return self.finish_reason in ("tool_calls", "stop")

@dataclass
class ToolResult:
    """ 工具执行结果 """
    id: str
    name: str
    # 工具执行的返回内容
    content: str | None
    # 执行耗时
    latency_ms: int = 0
    # 异常信息
    error: str | None = None

    def to_openai_tool_message(self) -> dict[str, Any] :
        tool_result_message = {
            "role": "tool",
            "id": self.id,
            "name": self.name,
            "content": self.content
        }

        return tool_result_message