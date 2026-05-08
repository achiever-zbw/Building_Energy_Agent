import asyncio
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.agent.tools.base import Tool
from backend.agent.tools.registry import ToolRegistry


class EchoTool(Tool):
    @property
    def name(self) -> str:
        return "echo_tool"

    @property
    def description(self) -> str:
        return "回显输入参数"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "count": {"type": "integer", "minimum": 1},
            },
            "required": ["text"],
        }

    async def execute(self, **kwargs: Any) -> Any:
        text = kwargs.get("text", "")
        count = kwargs.get("count", 1)
        return " ".join([text] * count)


class ErrorTool(Tool):
    @property
    def name(self) -> str:
        return "error_tool"

    @property
    def description(self) -> str:
        return "模拟执行失败"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> Any:
        raise RuntimeError("mock failure")


def _build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(ErrorTool())
    return registry


def test_register_and_lookup():
    registry = _build_registry()
    assert registry.has("echo_tool")
    assert "error_tool" in registry
    assert len(registry) == 2
    assert registry.get("echo_tool") is not None


def test_prepare_call_cast_success():
    registry = _build_registry()
    tool, cast_params, err = registry.prepare_call("echo_tool", {"text": "hi", "count": "2"})
    assert err is None
    assert tool is not None and tool.name == "echo_tool"
    assert cast_params["count"] == 2


def test_prepare_call_missing_tool():
    registry = _build_registry()
    _, _, err = registry.prepare_call("missing_tool", {"x": 1})
    assert err is not None
    assert "未发现" in err


def test_prepare_call_invalid_params():
    registry = _build_registry()
    _, _, err = registry.prepare_call("echo_tool", {"count": 3})
    assert err is not None
    assert "非法参数" in err


def test_execute_success():
    registry = _build_registry()
    result = asyncio.run(registry.execute("echo_tool", {"text": "ok", "count": "3"}))
    assert result == "ok ok ok"


def test_execute_runtime_error():
    registry = _build_registry()
    result = asyncio.run(registry.execute("error_tool", {}))
    assert isinstance(result, str)
    assert "发生错误" in result


def test_execute_error_prefixed_result_adds_hint():
    class ErrorPrefixTool(Tool):
        @property
        def name(self) -> str:
            return "error_prefix_tool"

        @property
        def description(self) -> str:
            return "返回 Error 前缀文本"

        @property
        def parameters(self) -> dict[str, Any]:
            return {"type": "object", "properties": {}}

        async def execute(self, **kwargs: Any) -> Any:
            return "Error: bad args"

    registry = ToolRegistry()
    registry.register(ErrorPrefixTool())
    result = asyncio.run(registry.execute("error_prefix_tool", {}))
    assert "Error: bad args" in result
    assert "尝试另一种方法" in result


def test_get_definitions_cache_hit_and_invalidation():
    registry = _build_registry()
    d1 = registry.get_definitions()
    d2 = registry.get_definitions()
    assert d1 is d2

    class TmpTool(EchoTool):
        @property
        def name(self) -> str:
            return "tmp_tool"

    registry.register(TmpTool())
    assert registry._cached_definitions is None
    _ = registry.get_definitions()
    registry.unregister("tmp_tool")
    assert registry._cached_definitions is None


def test_get_definitions_builtins_before_mcp():
    class MCPTool(EchoTool):
        @property
        def name(self) -> str:
            return "mcp_fetch"

    registry = ToolRegistry()
    registry.register(MCPTool())
    registry.register(EchoTool())
    names = [x["function"]["name"] for x in registry.get_definitions()]
    assert names == ["echo_tool", "mcp_fetch"]
