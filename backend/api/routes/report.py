from fastapi import APIRouter, Body
from services.report import ReportService
from schemas.report import ExportReportRequest
from fastapi import HTTPException
from fastapi.responses import Response

router = APIRouter(
    prefix="/report",
    tags=["report"],
)

# 媒体类型和文件扩展名
_MEDIA = {
    "csv": "text/csv; charset=utf-8",
    "excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "json": "application/json; charset=utf-8",
}
_EXT = {"csv": "csv", "excel": "xlsx", "json": "json"}

@router.post("/export", summary="导出统计报表")
async def export_report(request: ExportReportRequest = Body(...)):
    try:
        body = await ReportService.report_file(
            request.data , file_type=request.file_type
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    ext = _EXT[request.file_type]
    return Response(
        content=body,
        media_type=_MEDIA[request.file_type],
        headers={"Content-Disposition": f"attachment; filename=report.{ext}"},
    )
    