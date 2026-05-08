from datetime import date

from fastapi import APIRouter, Body, Query
from schemas.search import BuildingTimeEnergyRequest, BuildingYearEnergyRequest
from schemas.search import BuildingTimeEnergyByHourRequest
from schemas.search import BuildingAnomalyDetectRequest
from services.search import SearchService
from services.anomaly import AnomalyService
from services.cop import CopSimulationService

router = APIRouter(
    prefix="/search",
    tags=["search"],
)


@router.post("/building_time_energy", summary="建筑时间能耗查询")
async def get_building_time_energy(
    request: BuildingTimeEnergyRequest = Body(...),
) :
    return await SearchService.mcp_get_building_time_energy(
        request.building_id,
        request.start_time,
        request.end_time,
    )
    

@router.post("/building_year_energy", summary="建筑年能耗查询")
async def get_building_year_energy(
    request: BuildingYearEnergyRequest = Body(...),
):
    return await SearchService.mcp_calculate_energy_intensity_preyear(
        building_id=request.building_id,
        year=request.year,
    )

@router.post("/building_time_energy_by_hour", summary="按照小时的粒度进行建筑能耗查询")
async def get_building_time_energy_by_hour(
    request: BuildingTimeEnergyByHourRequest = Body(...),
):
    return await SearchService.get_building_time_energy_by_hour(
        request.building_id,
        request.start_time,
        request.end_time,
    )

@router.get("/all_building_id", summary="获取所有建筑的 id")
async def get_all_building_id():
    return await SearchService.get_all_building_id()


@router.post("/anomaly_detect", summary="建筑能耗异常分析")
async def anomaly_detect(
    request: BuildingAnomalyDetectRequest = Body(...),
):
    return await AnomalyService.analyze_building_energy_month_zscore(
        building_id=request.building_id,
        meter_type=request.meter_type,
        year=request.year,
        month=request.month,
    )


@router.get("/cop_simulation", summary="冷水机组 COP 模拟数据")
async def get_cop_simulation(
    building_id: str = Query("demo", description="建筑编号"),
    days: int = Query(7, ge=1, le=90, description="统计天数"),
    end_date: date | None = Query(None, description="结束日期，默认今天"),
):
    """
    返回按日汇总的模拟冷负荷、机组功率与 COP；同一建筑与同参下序列可复现。
    """
    return CopSimulationService.simulate(
        building_id=building_id.strip() or "demo",
        days=days,
        end_date=end_date,
    )