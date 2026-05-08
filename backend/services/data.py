# 对上传数据进行处理
from fastapi import HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models.energy import Energy
from models.building import Building
from models.weather import Weather
import pandas as pd
from utils.time import transform_timestamp

transform_bool = {
    "YES": True , "NO": False , 
    "true": True , "false": False , 
    "1": True , "0": False , 
    "Y": True , "N": False , 
    "T": True , "F": False , 
    "True": True , "False": False , 
}

class DataService:
    """ 数据处理服务 """
    @staticmethod
    async def process_upload_building_data(
        db: AsyncSession,
        file: UploadFile,
    ) -> dict:
        """ 处理上传建筑信息数据 """
        try:
            # 读取文件格式（UploadFile 底层文件指针可能不在开头）
            upload_file_path = file.file
            upload_file_path.seek(0)
            upload_file_type = (file.filename or "").split(".")[-1].lower()
            if upload_file_type not in ["csv", "json"]:
                raise HTTPException(
                    status_code=400,
                    detail="文件格式错误，请上传 csv 或 json 文件"
                )

            # 读取文件
            if upload_file_type == "csv":
                df = pd.read_csv(upload_file_path)
            elif upload_file_type == "json":
                df = pd.read_json(upload_file_path)

            # 逐个字段处理，加入数据库
            for index, row in df.iterrows():
                building_id = row.get("building_id")
                site_id = row.get("site_id")
                primaryspaceusage = row.get("primaryspaceusage")
                sqm = row.get("sqm")
                occupants = row.get("occupants")
                yearbuilt = row.get("yearbuilt")
                have_electricity = row.get("have_electricity")
                have_chilledwater = row.get("have_chilledwater")
                have_hotwater = row.get("have_hotwater")
                have_water = row.get("have_water")

                if pd.isna(building_id):
                    continue

                # 检查 building_id 是否已经存在
                sql = select(Building).where(Building.building_id == building_id)
                result = await db.execute(sql)
                exist_building = result.scalars().first()

                if exist_building:
                    continue

                building = Building(
                    building_id=building_id,
                    site_id=site_id,
                    primaryspaceusage=primaryspaceusage,
                    sqm=sqm,
                    occupants=occupants,
                    yearbuilt=yearbuilt,
                    have_electricity=have_electricity,
                    have_chilledwater=have_chilledwater,
                    have_hotwater=have_hotwater,
                    have_water=have_water,
                )
                
                db.add(building)

            await db.commit()

            return {
                "status": "success",
                "message": "建筑信息数据上传成功",
            }

        except HTTPException:
            raise
        except Exception as e:
            await db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"建筑信息数据上传失败: {e!s}",
            ) from e
    
    @staticmethod
    async def process_upload_energy_data(
        db: AsyncSession,
        file: UploadFile,
    ) -> dict:
        """ 处理上传的能耗数据 """
        try:
            upload_file_path = file.file
            upload_file_path.seek(0)
            upload_file_type = (file.filename or "").split(".")[-1].lower()
            if upload_file_type not in ["csv", "json"]:
                raise HTTPException(
                    status_code=400,
                    detail="文件格式错误，请上传 csv 或 json 文件"
                )

            # 读取文件
            if upload_file_type == "csv":
                df = pd.read_csv(upload_file_path)
            elif upload_file_type == "json":
                df = pd.read_json(upload_file_path)

            meter_types = ["electricity", "chilledwater", "hotwater", "water"]
            for index, row in df.iterrows():
                building_id = row.get("building_id")
                timestamp = row.get("timestamp")
                
                if pd.isna(building_id) or pd.isna(timestamp):
                    continue

                sql = select(Building).where(Building.building_id == building_id)
                result = await db.execute(sql)
                exist_building = result.scalars().first()
                if not exist_building:
                    continue

                for meter_type in meter_types:
                    if meter_type not in row:
                        continue
                    meter_value = row[meter_type]
                    # 如果是NaN ，跳过，数据表中 value 是允许有空值的
                    if pd.isna(meter_value):
                        continue

                    ts = transform_timestamp(timestamp)

                    # 检查唯一约束
                    sql_dup = select(Energy).where(
                        Energy.building_id == building_id,
                        Energy.timestamp == ts,
                        Energy.meter_type == meter_type,
                    )
                    res_dup = await db.execute(sql_dup)
                    if res_dup.scalars().first() is not None:
                        continue

                    energy_record = Energy(
                        building_id=building_id,
                        timestamp=ts,
                        meter_type=meter_type,
                        meter_value=float(meter_value),
                    )
                    db.add(energy_record)

            await db.commit()

            return {
                "status": "success",
                "message": "能耗数据上传成功",
            }
        except HTTPException:
            raise

        except Exception as e:
            import traceback
            traceback.print_exc()
            await db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"能耗数据上传失败: {e!s}",
            ) from e

    @staticmethod
    async def process_upload_weather_data(
        db: AsyncSession,
        file: UploadFile,
    ) -> dict:
        """ 处理上传的天气数据 """
        try:
            upload_file_path = file.file
            upload_file_path.seek(0)
            upload_file_type = (file.filename or "").split(".")[-1].lower()
            if upload_file_type not in ["csv", "json"]:
                raise HTTPException(
                    status_code=400,
                    detail="文件格式错误，请上传 csv 或 json 文件"
                )

            # 读取文件
            if upload_file_type == "csv":
                df = pd.read_csv(upload_file_path)
            elif upload_file_type == "json":
                df = pd.read_json(upload_file_path)

            for index, row in df.iterrows():
                site_id = row.get("site_id")
                timestamp = row.get("timestamp")
                if pd.isna(site_id) or pd.isna(timestamp):
                    continue
                
                # 只检查必须字段（site_id, timestamp）
                air_temperature = row.get("air_temperature")
                if pd.isna(air_temperature):
                    continue
                dew_temperature = row.get("dew_temperature")
                if pd.isna(dew_temperature):
                    continue
                wind_speed = row.get("wind_speed")
                if pd.isna(wind_speed):
                    continue
                cloud_coverage = row.get("cloud_coverage")
                if pd.isna(cloud_coverage):
                    continue
                
                # 降水量允许为空（改为 None）
                precipitation = row.get("precipitation")
                if pd.isna(precipitation):
                    precipitation = None  # 关键修改：将 NaN 转为 None（SQL NULL）
                else:
                    precipitation = float(precipitation)

                ts = transform_timestamp(timestamp)

                # 至少有一栋建筑使用该 site_id 时再入库
                sql_site = select(Building).where(Building.site_id == site_id)
                res_site = await db.execute(sql_site)
                if res_site.scalars().first() is None:
                    continue

                # 检查唯一约束
                sql_dup = select(Weather).where(
                    Weather.site_id == site_id,
                    Weather.timestamp == ts,
                )
                res_dup = await db.execute(sql_dup)
                if res_dup.scalars().first() is not None:
                    continue

                weather_record = Weather(
                    site_id=site_id,
                    timestamp=ts,
                    air_temperature=float(air_temperature),
                    dew_temperature=float(dew_temperature),
                    wind_speed=float(wind_speed),
                    cloud_coverage=float(cloud_coverage),
                    precipitation=precipitation,  # 可能是 None
                )
                db.add(weather_record)

            await db.commit()

            return {
                "status": "success",
                "message": "天气数据上传成功",
            }
        except HTTPException:
            raise
        except Exception as e:
            await db.rollback()
            raise HTTPException(
                status_code=500,
                detail=f"天气数据上传失败: {e!s}",
            ) from e