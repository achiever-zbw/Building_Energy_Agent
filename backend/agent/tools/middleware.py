# 中间件
from langchain.agents import AgentState
from langgraph.types import Command
from typing import Callable
from langchain.agents.middleware import AgentMiddleware, wrap_tool_call, before_model
from langgraph.prebuilt.tool_node import ToolCallRequest
from backend.utils.logger_handler import logger
from langgraph.runtime import Runtime

# 日志监控
@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,
        # 执行的函数本身
        handler: Callable[[ToolCallRequest], ToolCallRequest | Command],
) -> ToolCallRequest | Command:
    # 记录日志
    logger.info(f"[tool monitor]执行工具 : {request.tool_call["name"]}")
    logger.info(f"[tool monitor]传入参数 : {request.tool_call["args"]}")

    try:
        result = handler(request)
        logger.info(f"[tool monitor] 工具 : {request.tool_call["name"]} 调用成功")
        return result
    except Exception as e:
        logger.error(f"[tool monitor] 工具 : {request.tool_call["name"]} 调用失败")
        raise e

# 模型监控
@before_model
def log_before_model(
        state: AgentState,      # 整个 Agent 智能体的状态记录
        runtime: Runtime        # 记录执行过程的上下文信息
):
    logger.info(f"[log_before_model]即将调用模型 , 带有 {len(state["messages"])} 条信息。")
    logger.debug(f"[log_before_model] {type(state["messages"][-1].__name__)} | {state['messages'][-1].content.strip()}")

    return None

