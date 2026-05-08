import json
import os
from typing import Any
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)
from loguru import logger
from backend.agent.schemas import LLMResponse, ToolCallRequest
from backend.agent.providers.base import LLMProvider
import asyncio

# Default OpenAI HTTP 最大响应时间
_OPENAI_REQUEST_TIMEOUT_S = 120.0

def _openai_timeout_s() -> float:
    """ 返回 HTTP 最大等待时间 """
    return _float_env("OPENAI_TIMEOUT_S", _OPENAI_REQUEST_TIMEOUT_S)


def _float_env(name: str, default: float) -> float:
    """ 安全解析浮点数 ，无效值时回退到默认值并记录警告"""
    raw = os.environ.get(name)
    # 如果不存在，采用默认值
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("缺少环境配置 {}={!r}; 使用 {}", name, raw, default)
        return default
    return value

class OpenAIProvider(LLMProvider):
    """ OpenAI 风格 API 客户端，继承 LLMProvider，封装与 OpenAI SDK 的交互 """
    def __init__(
            self,
            api_key: str | None = None,
            api_base: str | None = None,
            default_model: str  = "deepseek-chat",
    ):
        super().__init__(api_key, api_base)
        self.api_key = api_key
        self.api_base = api_base
        self.default_model = default_model

        # 获取最大响应时间
        timeout_s = _openai_timeout_s()
        # 构建客户端
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            # 禁用自带重试，retry次数设置为 0
            max_retries=0,
            timeout=timeout_s,
        )

    # ------------------------------------------------------------------
    # Build kwargs
    # ------------------------------------------------------------------

    @staticmethod
    def _supports_temperature(
            model_name: str,
            reasoning_effort: str | None = None,
    ) -> bool:
        """返回模型是否支持温度参数

        GPT-5 family and reasoning models (o1/o3/o4) reject temperature
        when reasoning_effort is set to anything other than ``"none"``.
        """
        if reasoning_effort and reasoning_effort.lower() != "none":
            return False
        name = model_name.lower()
        return not any(token in name for token in ("gpt-5", "o1", "o3", "o4"))

    def _build_kwargs(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None,
            model: str | None,
            max_tokens: int,
            temperature: float,
            reasoning_effort: str | None,
            tool_choice: str | dict[str, Any] | None,
    ) -> dict[str, Any]:
        """ 构建业务参数 -> OpenAI 接受的字典参数 """
        model_name = model or self.default_model
        # 显式判 None：避免 max_tokens=0 这种边缘情况被 `or` 错误兜底
        effective_max_tokens = (
            max_tokens if max_tokens is not None else self.generation.max_tokens
        )
        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max(1, effective_max_tokens),
        }
        # 只有提供了工具再添加 tool
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        # temperature 参数由动态决定
        if self._supports_temperature(model_name, reasoning_effort):
            kwargs["temperature"] = temperature

        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort

        return kwargs

    @classmethod
    def _extract_usage(cls, response: Any) -> dict[str, int]:
        """提取 token 使用量，只处理常见 OpenAI 格式。"""

        # 1. 获取 usage 对象/字典
        usage = None
        if isinstance(response, dict):
            usage = response.get("usage")
        elif hasattr(response, "usage") and response.usage:
            usage = response.usage

        if not usage:
            return {}

        # 2. 提取三个基本字段（兼容 dict 和对象）prompt_tokens, completion_tokens, total_tokens
        if isinstance(usage, dict):
            prompt = int(usage.get("prompt_tokens", 0))
            completion = int(usage.get("completion_tokens", 0))
            total = int(usage.get("total_tokens", 0))
        else:
            prompt = getattr(usage, "prompt_tokens", 0) or 0
            completion = getattr(usage, "completion_tokens", 0) or 0
            total = getattr(usage, "total_tokens", 0) or 0

        result = {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": total,
        }

        # 3. 尝试获取 cached_tokens（仅当顶层存在或简单嵌套）
        cached = None
        if isinstance(usage, dict):
            # 尝试顶层 cached_tokens
            if "cached_tokens" in usage:
                cached = usage["cached_tokens"]
            # 尝试 prompt_tokens_details.cached_tokens
            elif "prompt_tokens_details" in usage:
                details = usage["prompt_tokens_details"]
                if isinstance(details, dict):
                    cached = details.get("cached_tokens")
        else:
            # 对象属性
            if hasattr(usage, "cached_tokens"):
                cached = usage.cached_tokens
            elif hasattr(usage, "prompt_tokens_details"):
                details = usage.prompt_tokens_details
                if hasattr(details, "cached_tokens"):
                    cached = details.cached_tokens

        if cached is not None:
            result["cached_tokens"] = int(cached)

        return result

    def _parse(self, response: Any) -> LLMResponse:
        """ 响应解析：只处理 OpenAI SDK 对象和普通 dict，忽略复杂兼容逻辑。"""

        # 1.字符串直接返回
        if isinstance(response, str):
            return LLMResponse(content=response, finish_reason="stop")

        # 2.处理字典
        if isinstance(response, dict):
            choices = response.get("choices", [])
            if not choices:
                # 没有 choices，尝试顶层字段
                content = response.get("content") or response.get("output_text")
                return LLMResponse(
                    content=content,
                    finish_reason=str(response.get("finish_reason", "stop")),
                    usage=self._extract_usage(response),
                )
            choice = choices[0]
            message = choice.get("message", {})
            content = message.get("content")
            finish_reason = choice.get("finish_reason") or "stop"
            reasoning_content = message.get("reasoning_content")

            # 提取 tool_calls
            tool_calls = []
            raw_tcs = message.get("tool_calls", []) or []
            for tc in raw_tcs:
                # 获取函数名称、参数
                func = tc.get("function", {})
                args = func.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                # 使用 ToolCallRequest 封装提取到的各项参数
                tool_calls.append(ToolCallRequest(
                    # 唯一 id
                    id=tc.get("id", ""),
                    name=func.get("name", ""),
                    arguments=args if isinstance(args, dict) else {},
                ))

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=self._extract_usage(response),
                reasoning_content=reasoning_content,
            )

        # 3.处理 OpenAI SDK 对象（有 choices 属性）
        if hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            message = choice.message
            content = message.content
            finish_reason = choice.finish_reason or "stop"

            tool_calls = []
            if hasattr(message, "tool_calls") and message.tool_calls:
                for tc in message.tool_calls:
                    args = tc.function.arguments
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except json.JSONDecodeError:
                            args = {}
                    tool_calls.append(ToolCallRequest(
                        id=tc.id,
                        name=tc.function.name,
                        arguments=args if isinstance(args, dict) else {},
                    ))

            reasoning_content = getattr(message, "reasoning_content", None)
            usage = self._extract_usage(response)

            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
                usage=usage,
                reasoning_content=reasoning_content,
            )

        # 兜底：无法解析
        return LLMResponse(content="错误: 无法解析响应体", finish_reason="error")


    async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            model: str | None = None,
            max_tokens: int | None = None,
            temperature: float = 0.2,
            reasoning_effort: str | None = None,
            tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """ 底层 LLM 调用接口
        """
        kwargs = self._build_kwargs(
            messages=messages,
            tools=tools,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        )
        try:
            result = await self._client.chat.completions.create(**kwargs)
            return self._parse(result)
        except Exception as e:
            return self._handle_error(e)

    def get_default_model(self) -> str:
        return self.default_model

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    def _handle_error(self, e: Exception) -> LLMResponse:
        """ 把异常包成 LLMResponse(error)，并提取结构化元数据
        让父类 `_is_transient_response` 能基于 status_code / kind / retry_after
        命中前 3 级判断，而不是只能走文本匹配兜底。
        将任何调用 LLM API 的错误信息，转化为标准的 LLMResponse 对象进行封装
        """
        status_code: int | None = None
        error_kind: str | None = None
        error_type: str | None = None
        error_code: str | None = None
        retry_after: float | None = None
        should_retry: bool | None = None
        # 请求超时
        if isinstance(e, APITimeoutError):
            error_kind = "timeout"
        # 连接错误
        elif isinstance(e, APIConnectionError):
            error_kind = "connection"
        # HTTP 状态码错误
        if isinstance(e, APIStatusError):
            status_code = e.status_code
            response = getattr(e, "response", None)
            if response is not None and hasattr(response, "headers"):
                retry_after = self._extract_retry_after(response.headers)
                should_retry = self._extract_should_retry(response.headers)


            """
            # type / code 兜底：
            # 1.OpenAI SDK 1.x 在异常对象上直接暴露 e.type / e.code
            # 2.e.body = {"error": {"type": "...", "code": "..."}}（带 error 包装）
            # 3) e.body = {"type": "...", "code": "..."}（已被 SDK 剥壳）
            {
                'error': 
                    {
                        'message': 'Authentication Fails, Your api key: sk- is invalid', 
                        'type': 'authentication_error', 
                        'param': None, 
                        'code': 'invalid_request_error'
                    }
            }
            """
            error_type = getattr(e, "type", None)
            error_code = getattr(e, "code", None)

            if error_type is None or error_code is None:
                body = getattr(e, "body", None)
                if isinstance(body, dict):
                    inner = body.get("error")
                    err_dict = inner if isinstance(inner, dict) else body
                    if isinstance(err_dict, dict):
                        error_type = error_type or err_dict.get("type")     # 错误类型提取，e.g.'authentication_error'
                        error_code = error_code or err_dict.get("code")     # 错误码 e.g. 'invalid_request_error'
        else:
            # 兜底：非 OpenAI SDK 异常也尝试取 status_code
            status_code = getattr(e, "status_code", None)

        logger.warning(
            "LLM 调用失败 status_code={} kind={} type={} code={} err={}",
            status_code, error_kind, error_type, error_code, e,
        )

        return LLMResponse(
            content=f"调用 LLM 出错: {e}",
            finish_reason="error",
            error_status_code=status_code,
            error_kind=error_kind,
            error_type=error_type,
            error_code=error_code,
            error_retry_after_s=retry_after,
            error_should_retry=should_retry,
        )

    def _extract_retry_after(self, headers: Any) -> float | None:
        """ 从 HTTP 相应头中提取 Retry-After 字段的值，转化为秒数 """
        if headers is None:
            return None
        raw = headers.get("retry-after") if hasattr(headers, "get") else None
        if not raw:
            return None
        try:
            return float(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_should_retry(headers: Any) -> bool | None:
        """ 从 HTTP 响应头提取 x-should-retry 字段
        部分厂商 (Anthropic / OpenRouter) 通过该字段显式告知客户端是否应该重试，
        优先级高于客户端基于 status_code 的猜测。
        DeepSeek / Qwen 不使用该字段，对应返回 None。
        """
        if headers is None:
            return None
        raw = headers.get("x-should-retry") if hasattr(headers, "get") else None
        if not isinstance(raw, str):
            return None
        lowered = raw.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        return None


# test
async def main():
    # provider = OpenAIProvider(
    #     api_key="sk-bfe5bcf94a2a4d039de41c7369d37d72",
    #     api_base="https://api.deepseek.com",
    #     default_model="deepseek-chat",
    # )
    #
    # print("1. 普通对话 \n")
    # messages = [{"role": "user", "content": "你好，你是什么模型?"}]
    # r = await provider.chat_with_retry(messages)
    #
    # print("content: " , r.content)
    # print("finish: " , r.finish_reason)
    # print("usage: " , r.usage)
    # print("reasoning_content: " , r.reasoning_content)
    # print("tool_calls: " , r.tool_calls)
    # print("error_status_code: " , r.error_status_code)
    # print("error_kind: " , r.error_kind)
    # print("error_type: " , r.error_type)
    # print("error_code: " , r.error_code)
    # print("error_retry_after_s: " , r.error_retry_after_s)
    #
    # print("\n2. 工具测试")
    # tools = [{
    #     "type": "function",
    #     "function":{
    #         "name": "get_weather",
    #         "description": "获取天气",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "city": {"type": "string", "description": "城市名称"}
    #             },
    #             "required": ["city"]
    #         }
    #     }
    # }]
    # messages = [{"role": "user", "content": "今天北京天气如何?"}]
    # r = await provider.chat_with_retry(messages, tools=tools)
    #
    # print("content: " , r.content)
    # print("finish: " , r.finish_reason)
    # print("usage: " , r.usage)
    # print("reasoning_content: " , r.reasoning_content)
    # print("tool_calls: " , r.tool_calls)
    # print("error_status_code: " , r.error_status_code)
    # print("error_kind: " , r.error_kind)
    # print("error_type: " , r.error_type)
    # print("error_code: " , r.error_code)
    # print("error_retry_after_s: " , r.error_retry_after_s)

    print("\n3. 错误测试")
    bad_provider = OpenAIProvider(
        api_key="sk-",
        api_base="https://api.deepseek.com",
        default_model="deepseek-chat",
    )
    r = await bad_provider.chat_with_retry([{"role": "user", "content": "你好，你是什么模型?"}])
    print(r)

if __name__ == '__main__':
    asyncio.run(main())