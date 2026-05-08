import asyncio
import sys
from pathlib import Path
from typing import Any

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from backend.agent.tools.base import Schema, Tool, tool_parameters


class IntRangeSchema(Schema):
    def to_json_schema(self) -> dict[str, Any]:
        return {"type": "integer", "minimum": 1, "maximum": 10}


class DummyTool(Tool):
    @property
    def name(self) -> str:
        return "dummy_tool"

    @property
    def description(self) -> str:
        return "dummy tool for tests"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "minimum": 0, "maximum": 120},
                "name": {"type": "string", "minLength": 1},
                "is_admin": {"type": "boolean"},
                "scores": {"type": "array", "items": {"type": "number"}, "minItems": 1},
                "meta": {
                    "type": "object",
                    "properties": {"level": {"type": "integer"}},
                    "required": ["level"],
                },
            },
            "required": ["age", "name"],
        }

    async def execute(self, **kwargs: Any) -> Any:
        return kwargs


class NonObjectSchemaTool(Tool):
    @property
    def name(self) -> str:
        return "bad_schema_tool"

    @property
    def description(self) -> str:
        return "schema type is not object"

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "string"}

    async def execute(self, **kwargs: Any) -> Any:
        return kwargs


def test_resolve_json_schema_type_union():
    assert Schema.resolve_json_schema_type(["string", "null"]) == "string"
    assert Schema.resolve_json_schema_type("integer") == "integer"


def test_subpath_building():
    assert Schema.subpath("", "age") == "age"
    assert Schema.subpath("user", "age") == "user.age"


def test_validate_integer_rejects_bool():
    errs = Schema.validate_json_schema_value(True, {"type": "integer"}, "age")
    assert errs and "integer" in errs[0]


def test_validate_number_rejects_bool():
    errs = Schema.validate_json_schema_value(False, {"type": "number"}, "score")
    assert errs and "number" in errs[0]


def test_validate_nullable_accepts_none():
    schema = {"type": ["string", "null"]}
    assert Schema.validate_json_schema_value(None, schema, "name") == []


def test_validate_enum_min_max_and_length_constraints():
    schema = {
        "type": "string",
        "enum": ["a", "b"],
        "minLength": 2,
        "maxLength": 3,
    }
    errs = Schema.validate_json_schema_value("c", schema, "code")
    assert any("必须是" in e for e in errs)
    assert any("至少为" in e for e in errs)


def test_validate_object_required_and_nested_path():
    schema = {
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {"age": {"type": "integer"}},
                "required": ["age"],
            }
        },
        "required": ["user"],
    }
    errs = Schema.validate_json_schema_value({"user": {"age": "18"}}, schema, "")
    assert any("user.age" in e for e in errs)


def test_validate_array_items_with_index_path():
    schema = {"type": "array", "items": {"type": "integer"}, "minItems": 1}
    errs = Schema.validate_json_schema_value([1, "x"], schema, "nums")
    assert any("nums[1]" in e for e in errs)


def test_validate_array_size_constraints():
    schema = {"type": "array", "minItems": 2, "maxItems": 3}
    errs = Schema.validate_json_schema_value([1], schema, "items")
    assert any("至少需要" in e for e in errs)


def test_schema_validate_value_uses_to_json_schema():
    schema = IntRangeSchema()
    assert schema.validate_value(5, "age") == []
    errs = schema.validate_value(20, "age")
    assert any("必须 <=" in e for e in errs)


def test_tool_cast_params_basic_conversion():
    tool = DummyTool()
    casted = tool.cast_params({"age": "18", "is_admin": "true", "name": 123})
    assert casted["age"] == 18
    assert casted["is_admin"] is True
    assert casted["name"] == "123"


def test_tool_cast_params_nested_object_and_array():
    tool = DummyTool()
    casted = tool.cast_params(
        {"age": "18", "name": "a", "scores": ["1.5", "2"], "meta": {"level": "3"}}
    )
    assert casted["scores"] == [1.5, 2.0]
    assert casted["meta"]["level"] == 3


def test_tool_validate_params_success_after_cast():
    tool = DummyTool()
    casted = tool.cast_params({"age": "20", "name": "tom", "scores": [1]})
    assert tool.validate_params(casted) == []


def test_tool_validate_params_missing_required():
    tool = DummyTool()
    errs = tool.validate_params({"name": "tom"})
    assert any("缺少必填字段 age" in e for e in errs)


def test_tool_validate_params_non_dict():
    tool = DummyTool()
    errs = tool.validate_params("bad")  # type: ignore[arg-type]
    assert errs and "参数必须是对象类型" in errs[0]


def test_tool_validate_params_schema_must_be_object():
    tool = NonObjectSchemaTool()
    with pytest.raises(ValueError):
        tool.validate_params({"x": 1})


def test_tool_to_schema_structure():
    tool = DummyTool()
    schema = tool.to_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "dummy_tool"
    assert schema["function"]["description"] == "dummy tool for tests"
    assert schema["function"]["parameters"]["type"] == "object"


def test_tool_execute_async():
    tool = DummyTool()
    result = asyncio.run(tool.execute(age=1, name="a"))
    assert result == {"age": 1, "name": "a"}


def test_tool_parameters_decorator_injects_parameters_property():
    @tool_parameters(
        {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }
    )
    class DecoratedTool(Tool):
        @property
        def name(self) -> str:
            return "decorated_tool"

        @property
        def description(self) -> str:
            return "decorated"

        async def execute(self, **kwargs: Any) -> Any:
            return kwargs

    tool = DecoratedTool()
    params = tool.parameters
    assert params["required"] == ["path"]
    params["required"].append("x")
    # shallow copy is expected in current backend implementation
    assert "x" in tool.parameters["required"]
