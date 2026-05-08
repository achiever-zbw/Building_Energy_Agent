""" 关于数据统计与查询的表单 """
from pydantic import BaseModel, Field
from datetime import datetime


class BuildingTimeEnergyRequest(BaseModel):
    """ 建筑时间能耗查询请求 """
    building_id: str = Field(..., description="建筑编号")
    start_time: datetime = Field(
        ...,
        description="开始时间（JSON 可用 ISO8601，如 2026-03-23T10:00:00）",
    )
    end_time: datetime = Field(
        ...,
        description="结束时间（JSON 可用 ISO8601）",
    )


class BuildingYearEnergyRequest(BaseModel):
    """ 建筑年能耗查询请求 """
    building_id: str = Field(..., description="建筑编号")
    year: int = Field(..., description="年份")


class BuildingTimeEnergyByHourRequest(BaseModel):
    """ 按照小时的粒度进行建筑能耗查询请求 """
    building_id: str = Field(..., description="建筑编号")
    start_time: datetime = Field(
        ...,
        description="开始时间（JSON 可用 ISO8601，如 2026-03-23T10:00:00）",
    )
    end_time: datetime = Field(
        ...,
        description="结束时间（JSON 可用 ISO8601）",
    )


class BuildingAnomalyDetectRequest(BaseModel):
    """ 建筑能耗异常分析请求 """
    building_id: str = Field(..., description="建筑编号")
    meter_type: str = Field(
        ...,
        description="能耗类型（如 electricity / chilledwater / hotwater / water）",
    )
    year: int = Field(..., description="年份，如 2016")
    month: int = Field(..., ge=1, le=12, description="月份，范围 1-12")