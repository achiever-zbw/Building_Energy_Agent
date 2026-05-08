from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List
from services.forecast import forecast_service   

router = APIRouter(prefix="/forecast", tags=["forecast"])

class ForecastData(BaseModel):
    timestamp: str
    predicted_energy: float

class ForecastResponse(BaseModel):
    status: str
    message: str
    building_id: str
    meter_type: str
    target_date: str
    forecast: List[ForecastData]

@router.get("/energy", response_model=ForecastResponse)
async def get_energy_forecast(
    building_id: str = Query(..., description="要预测的建筑 ID，例如 Moose_education_Lori"),
    meter_type: str = Query("electricity", description="能耗类型，例如 electricity"),
    target_date: str = Query(..., description="预测的目标日期 (起始零点)，例如 2016-04-10")
):
    """
    获取指定建筑在 target_date 的未来 24 小时能耗预测
    """
    try:
        results = await forecast_service.predict(building_id, meter_type, target_date)
        return ForecastResponse(
            status="success",
            message="预测成功",
            building_id=building_id,
            meter_type=meter_type,
            target_date=target_date,
            forecast=results
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"模型文件缺失: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"内部预测错误: {str(e)}")