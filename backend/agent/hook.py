""" Agent 运行生命周期 Hook  """
from dataclasses import dataclass, field
from typing import Any
from loguru import logger
from backend.agent.schemas import LLMResponse, ToolCallRequest

@dataclass(slots=True)
class AgentHookContext:
    """ 一些上下文信息
    把所有 Hook 可能需要的状态封装在里面，Hook 在执行一些操作时按需使用
    Example:
        - Token 计算 Hook 需要 usage 信息
    Runner 会不断修改、维护，Hook 进行读取，必要时进行修改
    """
    # 第几轮循环
    iteration: int
    # 完整的对话列表
    messages: list[dict[str, Any]]
    # 模型返回
    response: LLMResponse | None = None
    # token 用量
    usage: dict[str, int] = field(default_factory=dict)
    # 本次迭代需要执行的工具响应
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    # 执行结果
    tool_results: list[Any] = field(default_factory=list)
    # 工具执行事件
    tool_events: list[dict[str, str]] = field(default_factory=list)
    # 是否发生流式输出
    streamed_content: bool = False
    # 最终给用户的文本
    final_content: str | None = None
    # 停止原因
    stop_reason: str | None = None
    # 错误信息
    error: str | None = None


class AgentHook:
    """ Agent 调用时多个时机的挂载点 """
    def __init__(self, reraise: bool = False):
        # 该 Hook 是否向上抛出异常 (True: 报错 ; False: 异常捕捉并记录日志，不影响主流程)
        self._reraise = reraise

    def wants_streaming(self) -> bool:
        """Runner 在支持流式输出时可据此决定是否注册流式回调。"""
        return False

    async def before_iteration(self, context: AgentHookContext) -> None:
        pass

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        pass

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        pass

    async def after_iteration(self, context: AgentHookContext) -> None:
        pass

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        pass

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        return content

class CompositeHook(AgentHook):
    """ 将多个独立的 Hook 类组合成一个逻辑钩子，并按照顺序依次调用 """
    def __init__(self, hooks: list[AgentHook]) -> None:
        super().__init__()
        # 接受多个 Hook 组合成组合型 AgentHook
        self._hooks = list(hooks)

    def wants_streaming(self) -> bool:
        return any(h.wants_streaming() for h in self._hooks)

    async def _for_each_hook_safe(self, method_name: str, *args: Any, **kwargs: Any) -> None:
        """
        如果 CompositeHook 执行了某个时机(method_name)的方法，包含的子 Hook 依次执行该方法，sub_hook.method_name()
        """
        for h in self._hooks:
            # 遍历每个 Hook
            if getattr(h, "_reraise", False):
                # 如果 h._reraise == True(默认是 False) 表示子 Hook -> h 向上抛出异常，异常就崩掉
                await getattr(h, method_name)(*args, **kwargs)
                continue
            # 否则，通过 try except 来确保流程不会崩掉
            try:
                # 执行函数 h : Hook ; method_name: 方法名称  h.before_iteration
                await getattr(h, method_name)(*args, **kwargs)
            except Exception as e:
                logger.exception(f"AgentHook {method_name} failed, error: {str(e)}")


    async def before_iteration(self, context: AgentHookContext) -> None:
        """ 每次请求 LLM 之前挂载的钩子 """
        await self._for_each_hook_safe("before_iteration", context)

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        await self._for_each_hook_safe("on_stream", context, delta)

    async def on_stream_end(self, context: AgentHookContext, *, resuming: bool) -> None:
        await self._for_each_hook_safe("on_stream_end", context, resuming=resuming)

    async def after_iteration(self, context: AgentHookContext) -> None:
        """ 每次迭代完成进行的钩子 """
        await self._for_each_hook_safe("after_iteration", context)

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        """ 即将执行工具时挂载的钩子 """
        await self._for_each_hook_safe("before_execute_tools", context)

    def finalize_content(self, context: AgentHookContext, content: str | None) -> str | None:
        """ 对返回给用户的内容做整理，每个Hook对 content 进行修改，然后把所有 Hook 修改完后的 content 返回给用户 """
        for h in self._hooks:
            content = h.finalize_content(context, content)
        return content