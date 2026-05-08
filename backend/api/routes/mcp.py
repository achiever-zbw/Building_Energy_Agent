from fastapi import APIRouter, Request
from services.mcp import get_all_mcp_tools

router = APIRouter(
    prefix="/mcp",
    tags=["mcp"],
)

@router.get("/tools", summary="获取所有MCP工具")
async def get_all_mcp_tools_api(request: Request):
    return await get_all_mcp_tools(request)