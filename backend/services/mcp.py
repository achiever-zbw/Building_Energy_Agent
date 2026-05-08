from fastapi import HTTPException, Request

async def get_all_mcp_tools(
    request: Request,
):
    client = getattr(request.app.state, "mcp_client", None)
    if client is None or not getattr(client, "session", None):
        raise HTTPException(status_code=503, detail="MCP client 未初始化或未连接")

    try:
        response = await client.session.list_tools()
    except Exception as e:
        # 与前端联动：把“工具列表不可用”标为可恢复的服务依赖异常，而不是 500
        raise HTTPException(status_code=502, detail=f"获取 MCP 工具列表失败: {e!s}") from e

    tools = []

    for t in response.tools:
        tools.append({
            "mcp_tool": t.name , 
            "description": t.description , 
        })

    return tools