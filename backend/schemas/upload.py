# 上传文件的表单
from pydantic import BaseModel
from fastapi import UploadFile, File

class UploadFileForm(BaseModel):
    file_type: str    # 文件类型，buildings, energy, weather
    file: UploadFile = File(...)    # 上传文件路径