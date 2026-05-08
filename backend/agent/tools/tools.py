"""建筑能耗等业务工具：与 ``backend/mcp/server.py`` 中 MCP 工具同源，注册到 Agent ``ToolRegistry``。"""

from __future__ import annotations

from typing import Any

from backend.agent.tools.base import Tool, tool_parameters
from backend.agent.tools.registry import ToolRegistry
from backend.services.anomaly import AnomalyService
from backend.services.search import SearchService
from backend.utils.time import transform_timestamp


# --- get_building_time_energy -------------------------------------------------

_SCHEMA_GET_BUILDING_TIME_ENERGY = {
    "type": "object",
    "properties": {
        "building_id": {"type": "string", "description": "建筑 ID"},
        "start_time": {
            "type": "string",
            "description": "区间起始时间字符串（格式与同 MCP，将由 transform_timestamp 解析）",
        },
        "end_time": {
            "type": "string",
            "description": "区间结束时间字符串",
        },
    },
    "required": ["building_id", "start_time", "end_time"],
}


@tool_parameters(_SCHEMA_GET_BUILDING_TIME_ENERGY)
class GetBuildingTimeEnergyTool(Tool):
    concurrency_safe = True

    @property
    def name(self) -> str:
        return "get_building_time_energy"

    @property
    def description(self) -> str:
        return (
            "获取特定建筑在指定时间段内各能耗类型（电、冷冻水、热水、自来水）的累计能耗。"
        )

    async def execute(self, **kwargs: Any) -> Any:
        bid = kwargs["building_id"]
        st = transform_timestamp(kwargs["start_time"])
        et = transform_timestamp(kwargs["end_time"])
        return await SearchService.mcp_get_building_time_energy(bid, st, et)


# --- get_building_basic_info --------------------------------------------------

_SCHEMA_GET_BUILDING_BASIC_INFO = {
    "type": "object",
    "properties": {
        "building_id": {"type": "string", "description": "建筑 ID"},
    },
    "required": ["building_id"],
}


@tool_parameters(_SCHEMA_GET_BUILDING_BASIC_INFO)
class GetBuildingBasicInfoTool(Tool):
    concurrency_safe = True

    @property
    def name(self) -> str:
        return "get_building_basic_info"

    @property
    def description(self) -> str:
        return "查询建筑的元信息（用途类型、面积、建成年份、表计类型等）。"

    async def execute(self, **kwargs: Any) -> Any:
        return await SearchService.mcp_get_building_basic_info(kwargs["building_id"])


# --- calculate_energy_intensity_preyear ---------------------------------------

_SCHEMA_CALC_INTENSITY = {
    "type": "object",
    "properties": {
        "building_id": {"type": "string", "description": "建筑 ID"},
        "year": {"type": "integer", "description": "年份"},
    },
    "required": ["building_id", "year"],
}


@tool_parameters(_SCHEMA_CALC_INTENSITY)
class CalculateEnergyIntensityPreyearTool(Tool):
    concurrency_safe = True

    @property
    def name(self) -> str:
        return "calculate_energy_intensity_preyear"

    @property
    def description(self) -> str:
        return (
            "计算某建筑在指定自然年内各单位面积能耗（按能耗类型），用于能效对比。"
        )

    async def execute(self, **kwargs: Any) -> Any:
        return await SearchService.mcp_calculate_energy_intensity_preyear(
            kwargs["building_id"],
            int(kwargs["year"]),
        )


# --- anomaly_detect -----------------------------------------------------------

_SCHEMA_ANOMALY_DETECT = {
    "type": "object",
    "properties": {
        "building_id": {"type": "string"},
        "meter_type": {
            "type": "string",
            "description": "表计类型，如 electricity / chilledwater / hotwater / water",
        },
        "year": {"type": "integer"},
        "month": {"type": "integer", "minimum": 1, "maximum": 12},
    },
    "required": ["building_id", "meter_type", "year", "month"],
}


@tool_parameters(_SCHEMA_ANOMALY_DETECT)
class AnomalyDetectTool(Tool):
    concurrency_safe = True

    @property
    def name(self) -> str:
        return "anomaly_detect"

    @property
    def description(self) -> str:
        return (
            "对某建筑某能耗类型在指定年月的能耗序列做异常检测（Z-score 等），判断是否异常。"
        )

    async def execute(self, **kwargs: Any) -> Any:
        return await AnomalyService.analyze_building_energy_month_zscore(
            kwargs["building_id"],
            kwargs["meter_type"],
            int(kwargs["year"]),
            int(kwargs["month"]),
        )


# --- get_building_time_energy_by_hour -----------------------------------------

_SCHEMA_BY_HOUR = {
    "type": "object",
    "properties": {
        "building_id": {"type": "string"},
        "start_time": {"type": "string", "description": "区间起始时间字符串"},
        "end_time": {"type": "string", "description": "区间结束时间字符串"},
    },
    "required": ["building_id", "start_time", "end_time"],
}


@tool_parameters(_SCHEMA_BY_HOUR)
class GetBuildingTimeEnergyByHourTool(Tool):
    concurrency_safe = True

    @property
    def name(self) -> str:
        return "get_building_time_energy_by_hour"

    @property
    def description(self) -> str:
        return (
            "按小时粒度统计某建筑在时间区间内各类型能耗，便于画曲线或对比峰谷。"
        )

    async def execute(self, **kwargs: Any) -> Any:
        st = transform_timestamp(kwargs["start_time"])
        et = transform_timestamp(kwargs["end_time"])
        return await SearchService.get_building_time_energy_by_hour(
            kwargs["building_id"],
            st,
            et,
        )


_DEFAULT_TOOLS: tuple[type[Tool], ...] = (
    GetBuildingTimeEnergyTool,
    GetBuildingBasicInfoTool,
    CalculateEnergyIntensityPreyearTool,
    AnomalyDetectTool,
    GetBuildingTimeEnergyByHourTool,
)


def register_building_energy_tools(registry: ToolRegistry) -> None:
    """将建筑能耗相关 Tool 注册到已有 ``ToolRegistry``。"""
    for cls in _DEFAULT_TOOLS:
        registry.register(cls())


def build_default_tool_registry() -> ToolRegistry:
    """构造已注册全部默认业务工具的 ``ToolRegistry``（供 ``AgentLoop`` / ``main`` 使用）。"""
    reg = ToolRegistry()
    register_building_energy_tools(reg)
    return reg
