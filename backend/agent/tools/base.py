""" Base class for all tools """
from abc import ABC, abstractmethod
from typing import Any
from typing import Callable, TypeVar

_ToolT = TypeVar("_ToolT", bound="Tool")

# 数值类型 str -> type 的 map 映射
_JSON_TYPE_MAP : dict[str, type | tuple[type, ...]] = {
    "string": str,
    "number": (int, float),
    "integer": int,
    "boolean": bool,
    "array": list,
    "object": dict,
}

class Schema(ABC):
    """ JSON Schema 抽象基类，用于描述工具参数
    """
    @staticmethod
    def resolve_json_schema_type(t: Any) -> str | None:
        """ 从 JSON Schema 的 type 里拿到主类型，非 null """
        if isinstance(t, list):
            return next((x for x in t if x != "null"), None)
        return t

    @staticmethod
    def subpath(path: str, key: str) -> str:
        """ 拼接路径，方便观察 """
        return f"{path}.{key}" if path else key

    @staticmethod
    def validate_json_schema_value(val: Any, schema: dict[str, Any], path: str = "") -> list[str]:
        """ 核心验证逻辑 ， 递归验证每个值是否满足要求，返回验证错误的列表
        Args:
            val: 待验证的值
            schema: 约束表单
            path: e.g. user.age
        Returns:
            list: 错误列表

        Examples:
            1.  val = 25 ; schema = {"type": "integer", "minimum": 0, "maximum": 100} ; path = "age"
            2.  val = {"name": "张三", "age": 25 }
                schema = {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "age": {"type": "integer", "minimum": 0, "maximum": 150}
                    },
                    "required": ["name", "age"]
                }

        """
        # 获取 val 值的类型
        raw_type = schema.get("type")
        # nullable 字段是否允许为 null (1. type 是列表，里面有 null 字段 ["string", "null"]  2. schema 表单中有 "nullable" 的 key)
        nullable = (isinstance(raw_type, list) and "null" in raw_type) or schema.get("nullable", False)
        # 提取主类型
        t = Schema.resolve_json_schema_type(raw_type)
        label = path or "parameter"

        # 允许为空，直接通过验证
        if nullable and val is None:
            return []

        # 基础字段判断
        # 1. int
        if t == "integer" and (
            not isinstance(val, int) or isinstance(val, bool)
        ):
            return [f"{label} 应该是 integer 类型"]

        # 2. number
        if t == "number" and (
            not isinstance(val, _JSON_TYPE_MAP["number"]) or isinstance(val, bool)
        ):
            return [f"{label} 应该是 number 类型"]

        # 3. 其他类型，判断是否和 JSON MAP 的约定一致
        if t in _JSON_TYPE_MAP and t not in ("integer", "number") and not isinstance(val, _JSON_TYPE_MAP[t]):
            return [f"{label} 应该是 {t} 类型"]


        # 通用约束检查
        errors = []
        # 枚举
        if "enum" in schema and val not in schema["enum"]:
            errors.append(f"{label} 必须是 {schema['enum']} 中的一个")

        # 最大最小值比较
        if t in ("integer", "number"):
            if "minimum" in schema:
                if val < schema["minimum"]:
                    errors.append(f"{label} 必须 >= {schema['minimum']}")
            if "maximum" in schema:
                if val > schema["maximum"]:
                    errors.append(f"{label} 必须 <= {schema['maximum']}")

        # 检查 string 类型的 val 的长度是否符合最大最小长度
        if t == "string":
            if "minLength" in schema and len(val) < schema["minLength"]:
                errors.append(f"{label} 长度至少为 {schema['minLength']} 个字符")
            if "maxLength" in schema and len(val) > schema["maxLength"]:
                errors.append(f"{label} 长度最大为 {schema['maxLength']} 个字符")

        # 递归验证对象, val 中有多个子对象需要验证
        if t == "object":
            # 拿到每个对象的约束信息
            props = schema.get("properties", {})
            for k in schema.get("required", []):
                if k not in val:
                    errors.append(f"缺少必填字段 {Schema.subpath(path, k)}")
            # 对于每个子对象进行递归验证
            for k, v in val.items():
                if k in props:
                    # k: "name" , "age" ...
                    schema_k = props[k]     # key 对应的约束信息
                    val_k = v
                    errors.extend(Schema.validate_json_schema_value(val_k, schema_k, Schema.subpath(path, k)))



        # list 检查元素个数要求
        if t == "array":
            if "minItems" in schema and len(val) < schema["minItems"]:
                errors.append(f"{label} 至少需要 {schema['minItems']} 个元素")
            if "maxItems" in schema and len(val) > schema["maxItems"]:
                errors.append(f"{label} 最多只能有 {schema['maxItems']} 个元素")

            if "items" in schema:
                prefix = f"{path}[{{}}]" if path else "[{}]"
                # 列表子元素递归检测
                for i, item in enumerate(val):
                    error_message = Schema.validate_json_schema_value(item, schema["items"], prefix.format(i))
                    errors.extend(error_message)

        return errors

    @abstractmethod
    def to_json_schema(self) -> dict[str, Any]:
        """ 抽象方法，子类必须实现
        Returns:
            dict 自己的约束表单
        """
        ...


    def validate_value(self, value: Any, path: str = "") -> list[str]:
        """ 对自身参数的合法性判断
        Args:
            value: 待判定的值
            path: ""
        Returns:
            errors: list
        """
        # 首先要对本类的参数构建为合法的 schema 类型
        return Schema.validate_json_schema_value(value, self.to_json_schema(), path)

class Tool(ABC):
    """ Agent Tools 抽象基类 """

    _TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    # str 类型的 bool 值
    _BOOL_TRUE = frozenset(["true", "1", "yes"])
    _BOOL_FALSE = frozenset(["false", "0", "no"])

    @staticmethod
    def _resolve_type(t: Any) -> str | None:
        """ 提取主类型 """
        return Schema.resolve_json_schema_type(t)

    @property
    @abstractmethod
    def name(self) -> str:
        """ 抽象方法 : 定义工具名字"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """ 抽象方法 : 定义工具描述 """
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """ 抽象方法 : 定义工具参数 """
        ...


    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """ 抽象方法 : 定义工具实现方法，具体执行逻辑 """
        ...

    def _cast_value(self, val: Any, schema: dict[str, Any]) -> Any:
        """ 根据提供的 JSON Schema 对单个值进行安全的类型转换 """
        # 获取主类型
        t = self._resolve_type(schema.get("type"))

        # 如果 val 已经是正确的类型，就直接返回
        if t == "boolean" and isinstance(val, bool):
            return val
        if t == "integer" and isinstance(val, int) and not isinstance(val, bool):
            return val
        if t in self._TYPE_MAP and t not in ("boolean", "integer", "array", "object"):
            # 期望的正确对应类型
            excepted = self._TYPE_MAP[t]
            if isinstance(val, excepted):
                return val

        # "123", 数字可能以字符串的形式提供
        if isinstance(val, str) and t in ("integer", "number"):
            # 类型是数值，但是实际形式是 str， 要转换为数字
            try:
                return int(val) if t == "integer" else float(val)
            except ValueError:
                return val

        if t == "string":
            return val if val is None else str(val)

        # "false" "yes" "true" 等，str 类型的实际为 boolean 要转换为 bool
        if t == "boolean" and isinstance(val, str):
            low = val.lower()
            if low in self._BOOL_TRUE:
                # 在 true 的类型中
                return True
            if low in self._BOOL_FALSE:
                return False
            return val

        # 递归处理数组
        if t == "array" and isinstance(val, list):
            # items 是整个列表的 JSON Schema
            items = schema.get("items")
            # 每一项递归实现
            return [self._cast_value(x, items) for x in val] if items else val

        # dict object 类型，递归处理
        if t == "object" and isinstance(val, dict):
            # 调用对象处理函数进行处理
            return self._cast_object(val, schema)

        return val

    def _cast_object(self, obj: Any, schema: dict[str, Any]) -> Any:
        """ 对对象类型的 value 进行类型规范转换 """
        if not isinstance(obj, dict):
            return obj
        # 提取 properties, key 是属性名字 value 是该属性对应的 schema , 需要根据每个属性的表单进行 obj 中 value 的更新
        props = schema.get("properties", {})
        # 需要把对应的 value 用 _cast_value 转换器进行规范更改, props[k] 对应的是 key 的表单
        # _cast_value 需要的参数是待更行规范的 value 和 对应的 JSON Schema
        return {
            k: self._cast_value(v, props[k]) if k in props else v for k, v in obj.items()
        }

    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """ 对外提供参数的安全转换工具，把参数转换为标准形式 """
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            return params
        return self._cast_object(params, schema)

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """ 验证参数 """
        if not isinstance(params, dict):
            return [f"参数必须是对象类型 object , 实际类型为 {type(params).__name__}"]
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            raise ValueError(f"Schema 必须是对象类型, 实际类型为 {schema.get('type')!r}")
        return Schema.validate_json_schema_value(params, {**schema, "type": "object"}, "")

    def to_schema(self) -> dict[str, Any]:
        """ 将 tool 的信息标准化为 OpenAI 接受的 JSON 格式 """
        return {
            "type": "function",
            "function":{
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            }
        }

# 装饰器
def tool_parameters(schema: dict[str, Any]) -> Callable[[type[_ToolT]], type[_ToolT]]:
    """ 类装饰器 附加 JSON Schema 并注入具体的 parameters（参数）属性

    在 Tool 的子类上使用装饰器，装饰器自动把 schema 存储到类上

    Example::

        @tool_parameters({
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        })
        class ReadFileTool(Tool):
            ...
    """
    # cls：需要修饰的 Tool 类
    def decorator(cls: type[_ToolT]) -> type[_ToolT]:
        # 拷贝 schema
        frozen = dict(schema)

        @property
        def parameters(self: Any) -> dict[str, Any]:
            return dict(frozen)

        cls._tool_parameters_schema = dict(frozen)
        # 把 parameters 赋值给 Tool 类
        cls.parameters = parameters  # type: ignore[assignment]

        abstract = getattr(cls, "__abstractmethods__", None)
        if abstract is not None and "parameters" in abstract:
            cls.__abstractmethods__ = frozenset(abstract - {"parameters"})  # type: ignore[misc]
        # 返回赋值后的类
        return cls
    return decorator


if __name__ == '__main__':
    # test
    # 1
    # val = 120
    # schema = {"type": "integer", "minimum": 0, "maximum": 100}
    # path = "age"
    # result = Schema.validate_json_schema_value(val, schema, path)
    # print(result)
    #
    # # 2
    # val = {"name": "张三", "age": "13"}
    # schema = {
    #     "type": "object",
    #     "properties": {
    #         "name": {"type": "string", "minLength": 1},
    #         "age": {"type": "integer", "minimum": 0, "maximum": 100}
    #     },
    #     "required": ["name", "age"]
    # }
    # result = Schema.validate_json_schema_value(val, schema)
    # print(result)
    #
    #
    # # 3
    # class MySchema(Schema):
    #     def __init__(self, minimum=None, maximum=None):
    #         super().__init__()
    #         self.minimum = minimum
    #         self.maximum = maximum
    #
    #     def to_json_schema(self) -> dict[str, Any]:
    #         return {
    #             "type": "integer",
    #             "minimum": self.minimum,
    #             "maximum": self.maximum,
    #
    #         }
    #
    # obj = MySchema(minimum=20, maximum=100)
    #
    # errors = obj.validate_value(15, path="age")
    # print(errors)

    # 4.
    # 验证 Tool
    # class TestTool(Tool):
    #     @property
    #     def name(self) -> str:
    #         return "test_tool"
    #
    #     @property
    #     def description(self) -> str:
    #         return "这是用来测试的 test_tool"
    #
    #     @property
    #     def parameters(self) -> dict[str, Any]:
    #         return {
    #             "type": "object",
    #             "properties": {
    #                 "age": {"type": "integer", "minimum": 0, "maximum": 100},
    #                 "name": {"type": "string"},
    #                 "is_man": {"type": "boolean"},
    #             }
    #         }
    #
    #     async def execute(self, **kwargs) -> Any:
    #         print(kwargs)
    #
    # tool1 = TestTool()
    # params = {
    #     "age": "234",
    #     "ss": 2,
    # }
    # # tool 表单
    # tool_schema = tool1.to_schema()
    # for k, v in tool_schema.items():
    #     print(k, v)
    #
    # # 转换参数
    # casted_params = tool1.cast_params(params)
    # valid_messages = tool1.validate_params(casted_params)
    # print(valid_messages)

    # test 装饰器
    tool_schema = {
        "type": "object",
        "properties": {
            "age": {"type": "integer", "minimum": 0, "maximum": 100},
            "name": {"type": "string"},
            "is_man": {"type": "boolean"},
        }
    }

    @tool_parameters(tool_schema)
    class TestTool(Tool):
        @property
        def name(self) -> str:
            return "test_tool"

        @property
        def description(self) -> str:
            return "这是用来测试的 test_tool"

        async def execute(self, **kwargs) -> Any:
            print(kwargs)

        # 方法还是要实现，但是不需要具体去写了
        def parameters(self) -> dict[str, Any]:
            ...

    # 检查装饰器是否注入了参数
    test_tool = TestTool()
    print(test_tool.parameters)

    print(f"是否与原本的 schema 定义一致 : " , test_tool.parameters == tool_schema)