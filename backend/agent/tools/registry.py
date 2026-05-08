""" 工具注册 - 工具统一管理 """
from typing import Any
from backend.agent.tools.base import Tool


class ToolRegistry:
    """ 对 Agent Tools 进行统一注册管理
    支持工具的动态注册与执行
    """
    def __init__(self):
        # 存储工具
        self._tools : dict[str, Tool] = {}
        # 缓存，每次工具列表发生变动需要进行清空
        self._cached_definitions: list[dict[str, Any]] | None = None

    def register(self, tool: Tool) -> None:
        """ 注册工具
        Args:
             tool : Tool 工具对象
        """
        self._tools[tool.name] = tool
        # 工具列表变化，清空缓存
        self._cached_definitions = None

    def unregister(self, name: str) -> None:
        """ 删除工具
        Args:
            name: Tool.name
        """
        self._tools.pop(name, None)
        self._cached_definitions = None

    def get(self, name: str) -> Tool | None:
        """ 获取工具
        Args:
            name: Tool.name
        """
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """ 检查工具是否存在 """
        return name in self._tools

    @staticmethod
    def _schema_name(schema: dict[str, Any]) -> str:
        """ 从 Schema 表单中获取工具名字
        schema 支持两种形式 1. OpenAI 格式  2. 普通dict
        function : name , description, parameters
        """
        fn = schema.get("function")
        if isinstance(fn, dict):
            name = fn.get("name")
            if isinstance(name, str):
                return name
        name = schema.get("name")
        return name if isinstance(name, str) else ""

    def get_definitions(self) -> list[dict[str, Any]]:
        """ 获取已注册工具的定义表单，支持缓存友好的 prompt
        同一组工具的输出顺序一致，不受注册先后的影响
        工具顺序稳定，能够提高缓存命中率，降低成本和延迟
        """
        # 如果缓存非空，直接返回，避免重复计算
        if self._cached_definitions is not None:
            return self._cached_definitions

        definitions = [tool.to_schema() for tool in self._tools.values()]
        # 内置工具与 MCP 工具
        builtins: list[dict[str, Any]] = []
        mcp_tools: list[dict[str, Any]] = []
        for tool_schema in definitions:
            name = self._schema_name(tool_schema)
            if name.startswith("mcp_"):
                mcp_tools.append(tool_schema)
            else:
                builtins.append(tool_schema)
        # 根据工具名字字典序排序
        # 确保内置工具在mcp工具之前
        builtins.sort(key = self._schema_name)
        mcp_tools.sort(key = self._schema_name)
        self._cached_definitions = builtins + mcp_tools
        return self._cached_definitions

    def prepare_call(
            self, name: str,
            params: dict[str, Any]
    ) -> tuple[Tool | None, dict[str, Any], str | None]:
        """ 准备执行一个工具的准备工作
        1. 查找工具 2.转换参数 3.参数验证 4.返回结果
        Args:
            name: Tool.name
            params: 参数 dict
        Returns:
            tool: Tool | None
            dict[str, Any]: 标准化后的参数
            str: 错误信息
        """
        tool = self._tools.get(name)
        if not tool:
            return None, params, (
                f"错误 : Tool '{name}' 未发现，可获取的工具有 :  {', '.join(self.tool_names())}"
            )
        # 安全类型标准化、转换
        cast_params = tool.cast_params(params)
        # 参数验证
        errors = tool.validate_params(cast_params)
        if errors:
            return tool, cast_params, (
                f"错误 : 非法参数对于工具 '{name}': " + "; ".join(errors)
            )
        return tool, cast_params, None

    async def execute(self, name: str, params: dict[str, Any]) -> Any:
        """ 工具执行
        Args:
            name: Tool.name
            params: Tool.params
        Returns:
            Any: 工具执行结果
        1. 执行前首先调用准备函数 : prepare_call 获取 tool cast_params error
        2. 若存在 error 返回错误信息
        3. 执行工具
        4. 若工具执行存在错误，返回错误信息
        5. 顺利，返回执行结果
        """
        _HINT = "\n\n[分析错误信息，并尝试另一种方法]"
        # 获取工具
        tool, cast_params, errors = self.prepare_call(name, params)
        if errors:
            return errors

        try:
            assert tool is not None
            result = await tool.execute(**cast_params)
            if isinstance(result, str) and result.startswith("Error"):
                return result + _HINT
            return result
        except Exception as e:
            return f"执行工具 {name} 发生错误 : {str(e)}" + _HINT

    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self):
        return len(self.tool_names())

    def __contains__(self, name: str) -> bool:
        return name in self.tool_names()

if __name__ == "__main__":
    import asyncio

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

        async def execute(self, **kwargs) -> Any:
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

        async def execute(self, **kwargs) -> Any:
            raise RuntimeError("mock failure")

    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(ErrorTool())

    # 1) 基础注册检查
    print("工具列表:", registry.tool_names())
    assert registry.has("echo_tool")
    assert "error_tool" in registry
    assert len(registry) == 2

    # 2) prepare_call: 参数转换与校验
    tool, cast_params, err = registry.prepare_call("echo_tool", {"text": "hi", "count": "2"})
    assert err is None
    assert tool is not None and tool.name == "echo_tool"
    assert cast_params["count"] == 2

    # 3) prepare_call: 工具不存在
    _, _, err = registry.prepare_call("missing_tool", {"x": 1})
    assert err is not None and "未发现" in err

    # 4) prepare_call: 缺失必填参数
    _, _, err = registry.prepare_call("echo_tool", {"count": 3})
    assert err is not None and "非法参数" in err

    # 5) execute: 成功路径
    result = asyncio.run(registry.execute("echo_tool", {"text": "ok", "count": "3"}))
    print("echo_tool 执行结果:", result)
    assert result == "ok ok ok"

    # 6) execute: 工具内部抛异常
    result = asyncio.run(registry.execute("error_tool", {}))
    print("error_tool 执行结果:", result)
    assert isinstance(result, str) and "发生错误" in result

    # 7) 缓存行为：第一次构建，第二次命中缓存
    d1 = registry.get_definitions()
    d2 = registry.get_definitions()
    assert d1 is d2

    # 8) 注册/卸载后缓存失效
    class TmpTool(EchoTool):
        @property
        def name(self) -> str:
            return "tmp_tool"

    registry.register(TmpTool())
    assert registry._cached_definitions is None
    _ = registry.get_definitions()
    registry.unregister("tmp_tool")
    assert registry._cached_definitions is None

    print("registry.py 自测通过")
