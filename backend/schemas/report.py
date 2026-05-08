"""报表导出请求体"""
from typing import Literal

from pydantic import BaseModel, Field

ReportFileType = Literal["csv", "excel", "json"]


class ExportReportRequest(BaseModel):
    """导出：使用上一次查询返回的数据，后端不再查询数据库。"""

    data: dict = Field(..., description="查询接口返回的 JSON，原样放入此字段")
    file_type: ReportFileType = Field("csv", description="导出格式：csv / excel / json")
