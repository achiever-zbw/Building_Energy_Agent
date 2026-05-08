# 构建 ReAct 规范的 Agent
from langchain.agents import create_agent
from backend.mcp.client import mcp_client
from backend.agent.tools.middleware import *

class ReActAgent:
    def __init__(self):
        self.agent = create_agent(
            model = None,
            system_prompt = None,
            # 获取 mcp 工具
            tools = mcp_client.get_tools(),
            # 加载中间件
            middleware = [monitor_tool, log_before_model]
        )


    def execute_stream(self, query):
        input_dict = {
            "messages": [
                {
                    "role": "user",
                    "content": query
                }
            ]
        }

        for chunk in self.agent.stream(input_dict, stream_mode = "values") :
            latest_message = chunk["messages"][-1]  # 最后一条信息

            if latest_message.content:
                yield latest_message.content.strip() + "\n"