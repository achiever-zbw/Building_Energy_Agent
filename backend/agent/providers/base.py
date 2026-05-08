import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from collections.abc import Awaitable, Callable
from backend.agent.schemas import LLMResponse
from loguru import logger



@dataclass(frozen=True)
class GenerationSettings:
    """ 默认的生成参数 """
    # 温度设置
    temperature: float = 0.2
    # 最大 Token
    max_tokens: int = 4096
    # 一些模型支持的推理力度
    reasoning_effort: str | None = None


class LLMProvider(ABC):
    """
    LLM 抽象基类，子类需要实现 chat 以及 get_default_model
    """
    # 标准重试的延迟序列，第一次失败等待 1s, 以此类推
    _CHAT_RETRY_DELAYS = (1, 2, 4)

    # 文本标记列表，用于在相应中字符串匹配来进行错误识别
    _TRANSIENT_ERROR_MARKERS = (
        "429",
        "rate limit",
        "500",
        "502",
        "503",
        "504",
        "overloaded",
        "timeout",
        "timed out",
        "connection",
        "server error",
        "temporarily unavailable",
        "速率限制",
    )

    # HTTP 状态码 408: 请求超时 ; 429: 速率限制
    _RETRYABLE_STATUS_CODES = frozenset({408, 409, 429})

    # 结构化错误类型
    _TRANSIENT_ERROR_KINDS = frozenset({"timeout", "connection"})

    # 哨兵对象
    _SENTINEL = object()

    def __init__(self, api_key: str, api_base: str) -> None:
        self.api_key = api_key
        self.api_base = api_base
        self.generation: GenerationSettings = GenerationSettings()

    @staticmethod
    def _sanitize_empty_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """ 清洗消息内容，修复空字符串、空文本等字段
        将消息列表中 content 字段规范为 LLM API 能接受的格式，防止空字符串或者空列表等导致 API 调用失败
        Args:
            messages (list[dict[str, Any]])
        Returns:
            list[dict[str, Any]]

        """
        # LLM API 接受的格式 : list
        result : list[dict[str, Any]] = []
        for message in messages:
            # 获取 content
            content = message["content"]
            if content is None or (isinstance(content, str) and not content):
                # content 为空或者是空字符串
                clean = dict(message)
                # 当消息角色是 assistant 并且是纯工具调用，content 必须强制为 None
                # content 不允许是空字符串，所以以 "empty" 来代替比较合适
                clean["content"] = None if (message["role"] == "assistant" and message.get("tool_calls")) else "(empty)"
                result.append(clean)
                continue


            if isinstance(content, list):
                # 多模态或者分块内容
                """
                [
                    {"type": "text" , "text": "请描述这个图片"},
                    {"type": "image_url", "image_url": "xxx"}
                ]
                可能包含空文本块 {"text":  ""}
                可能包含内部调试字段 {"type": "text", "text": "hello", "_meta": {...}}
                """
                new_items: list[Any] = []
                changed = False     # Flag 标记

                # 遍历每个 item
                for item in content:
                    # text 字段为 空
                    if (
                            isinstance(item, dict)
                            and item.get("type") in ("text", "input_text", "output_text")
                            and not item.get("text")    # text 字段为空
                    ):
                        changed = True
                        # 跳过这个空文本块，没有意义
                        continue

                    # 元数据
                    # _meta 字段可能用于调试记录时间戳等内部信息，需要移除
                    if isinstance(item, dict) and "_meta" in item:
                        # {"type": "text", "text": "hello", "_meta": {...}}
                        # 把 key != "_meta" 的加入到里面
                        new_items.append({k: v for k, v in item.items() if k != "_meta"})
                        changed = True
                    else:
                        new_items.append(item)

                # 如果列表发生变化，有空块删除或者 _meta 被移除
                if changed:
                    clean = dict(message)
                    # new_item 里有有效的内容，使用更新的 content
                    if new_items:
                        clean["content"] = new_items
                    # 没有有效内容，并且消息角色是 assistant 并且是纯工具调用，content 必须强制为 None
                    elif message.get("role") == "assistant" and message.get("tool_calls"):
                        clean["content"] = None
                    # 其他情况，设置为 empty 即可
                    else:
                        clean["content"] = "(empty)"
                    result.append(clean)
                    continue

            # 如果传递了 dict，需要转化为 list 形式，即使只有一条
            if isinstance(content, dict):
                clean = dict(message)
                clean["content"] = [content]
                result.append(clean)
                continue

            result.append(message)
        return result

    # 抽象接口
    @abstractmethod
    async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
            model: str | None = None,
            max_tokens: int | None = None,
            temperature: float = 0.2,
            reasoning_effort: str | None = None,
            tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        pass

    @abstractmethod
    def get_default_model(self) -> str:
        pass

    async def _safe_chat(self, **kwargs) -> LLMResponse:
        """ 为 chat 提供异常安全包装
        chat 抛出的异常如果不中断，会直接终止 _run_with_retry 的重试循环，需要捕获到错误响应
        逻辑 : 每次上层调用 chat_with_retry, 每次循环进行 _safe_chat, 该方法能够获取响应，正确则结束循环，否则继续，
        因此 _safe_chat 需要具备给每次 chat 得到一个响应的功能
        """
        try:
            return await self.chat(**kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # 定义完成原因为 : error
            return LLMResponse(content=f"Error calling LLM : {exc}", finish_reason="error")



    async def chat_with_retry(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            model: str | None = None,
            max_tokens: object = _SENTINEL,
            temperature: object = _SENTINEL,
            reasoning_effort: object = _SENTINEL,
            tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """ chat 重试机制 """
        if max_tokens is self._SENTINEL or max_tokens is None:
            max_tokens = self.generation.max_tokens
        if temperature is self._SENTINEL or temperature is None:
            temperature = self.generation.temperature
        if reasoning_effort is self._SENTINEL:
            reasoning_effort = self.generation.reasoning_effort

        # 构建参数字典
        kwargs : dict[str, Any] = dict(
            messages = self._sanitize_empty_content(messages),
            tools = tools or [],
            model = model,
            max_tokens = max_tokens,
            temperature = temperature,
            reasoning_effort = reasoning_effort,
            tool_choice = tool_choice,
        )

        return await self._run_with_retry(
            self._safe_chat,
            kwargs,
        )

    @classmethod
    def _is_transient_error(cls, content: str | None) -> bool:
        # 基于文本内容的临时错误判断，检查错误消息的文本内容是否包含预定义的临时错误标记
        # 作为兜底方案，通过关键词匹配来判断是否应该重试
        err = (content or "").lower()
        return any(marker in err for marker in cls._TRANSIENT_ERROR_MARKERS)

    @classmethod
    def _is_transient_response(cls, response: LLMResponse) -> bool:
        """ 判断 LLM 相应是否为临时性错误，用于决定是否应该重试，永久性错误不会尝试 """
        # 1-优先级: Response.error_should_retry, 显式标记
        if response.error_should_retry is not None:
            return bool(response.error_should_retry)

        # 2-优先级 HTTP 状态码
        if response.error_status_code is not None:
            status = int(response.error_status_code)
            if status in cls._RETRYABLE_STATUS_CODES or status >= 500:
                return True
            return False
        # 3-优先级 error kind
        error_kind = response.error_kind
        if error_kind in cls._TRANSIENT_ERROR_KINDS:
            return True
        # 4-优先级 兜底方案，前面都没有检测到，使用文本内容匹配
        return cls._is_transient_error(response.content)


    async def _run_with_retry(
            self,
            call: Callable[..., Awaitable[LLMResponse]],
            kw: dict[str, Any] ,
    ) -> LLMResponse:
        """ 重试执行器，标准重试策略，最多重试固定次数，遇到非错误响应返回 """
        delays = list(self._CHAT_RETRY_DELAYS)
        last_response: LLMResponse | None = None

        # 三次尝试
        for attempt, delay in enumerate(delays, start=1):
            print(f"尝试: {attempt} / {len(delays)}")
            response = await call(**kw)
            last_response = response
            # 非 error 调用成功直接返回
            if response.finish_reason != "error":
                return response
            # 是 error 但是不是永久错误
            if not self._is_transient_response(response):
                return response

            # 临时错误，等待后重试
            logger.warning(
                f"LLM 临时错误，准备重试 (尝试 {attempt} / {len(delays)}): {(response.content or '')[:120]}"
            )
            # 延迟时间，优先看 error_retry_after_s 字段
            retry_delay = response.error_retry_after_s or response.retry_after or delay
            await asyncio.sleep(max(0.1, float(retry_delay)))

        # 返回最新的响应
        return last_response if last_response is not None else await call(**kw)