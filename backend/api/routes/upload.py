from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from api.depends import get_db
from services.data import DataService

router = APIRouter(
    prefix="/upload",
    tags=["upload"],
)

@router.post("/buildings" , summary="上传建筑数据")
async def upload_buildings(
    file: UploadFile = File(..., description="上传 CSV 或 JSON"),
    db: AsyncSession = Depends(get_db),
):
    return await DataService.process_upload_building_data(db, file)



@router.post("/energy" , summary="上传能耗数据")
async def upload_energy(
    file: UploadFile = File(..., description="上传 CSV 或 JSON"),
    db: AsyncSession = Depends(get_db),
):
    return await DataService.process_upload_energy_data(db, file)

@router.post("/weather" , summary="上传天气数据")
async def upload_weather(
    file: UploadFile = File(..., description="上传 CSV 或 JSON"),
    db: AsyncSession = Depends(get_db),
):
    return await DataService.process_upload_weather_data(db, file)


@router.post("/documents", summary="上传资料文档到知识库")
async def upload_documents(
    request: Request,
    files: list[UploadFile] = File(..., description="上传到 LightRAG（目录见环境变量 WORKING_DIR）"),
):
    svc = getattr(request.app.state, "lightrag_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="LightRAG 服务未挂载")
    return await svc.upload_document(files)
    