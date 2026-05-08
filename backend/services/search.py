from collections import defaultdict
from datetime import datetime
from sqlalchemy import select
from models.building import Building
from db.session import async_session
from models.building import Building
from models.energy import Energy
from utils.time import transform_timestamp


def _floor_hour(ts: datetime) -> datetime:
    """归一到整点，用于按小时分桶（跨天不会混到同一桶）。"""
    return ts.replace(minute=0, second=0, microsecond=0)

def add_all_data(data) -> dict :
    # 对于一个 building, 一段时间内对四种不同数据的求和
    electricity = sum(e.meter_value for e in data if e.meter_type == "electricity")
    chilledwater = sum(e.meter_value for e in data if e.meter_type == "chilledwater")
    hotwater = sum(e.meter_value for e in data if e.meter_type == "hotwater")
    water = sum(e.meter_value for e in data if e.meter_type == "water")
    return {
        "electricity" : electricity,
        "chilledwater" : chilledwater,
        "hotwater" : hotwater,
        "water" : water
    }

class SearchService:
    """ 综合查询统计服务 """
    @staticmethod
    async def mcp_get_building_time_energy(
        building_id: str,
        start_time: datetime,
        end_time: datetime,
    ):
        try:
            async with async_session() as session:
                st_time = start_time
                ed_time = end_time
                sql = select(Energy).where(
                    Energy.building_id == building_id, 
                    Energy.timestamp >= st_time, 
                    Energy.timestamp <= ed_time
                )
                result = await session.execute(sql)
                energy_data = result.scalars().all()
                all_data = {}
                all_data["building_id"] = building_id
                all_data["start_time"] = start_time
                all_data["end_time"] = end_time
                all_data.update(add_all_data(energy_data))
                
                return all_data

        except Exception as e:
            return {"error": "查询能耗数据失败", "detail": str(e)}

    @staticmethod
    async def mcp_get_building_basic_info(building_id: str):
        """ 查询某个特定建筑的元信息 """
        try:
            async with async_session() as session:
                sql = select(Building).where(Building.building_id == building_id)
                result = await session.execute(sql)
                building_data = result.scalars().first()
                if not building_data:
                    return {
                        "message": "没有找到建筑信息",
                    }
                return {
                    "building_id": building_data.building_id,
                    "site_id": building_data.site_id,
                    "primaryspaceusage": building_data.primaryspaceusage,
                    "sqm": building_data.sqm,
                    "occupants": building_data.occupants,
                    "yearbuilt": building_data.yearbuilt,
                    "have_electricity": building_data.have_electricity,
                    "have_chilledwater": building_data.have_chilledwater,
                    "have_hotwater": building_data.have_hotwater,
                    "have_water": building_data.have_water,
                }

        except Exception as e:
            return {"error": "查询建筑信息失败", "detail": str(e)}

    @staticmethod
    async def mcp_calculate_energy_intensity_preyear(
        building_id: str ,
        year: int,
    ):
        """ 计算某个特定建筑，在特定年份的单位面积能耗，可以用于比较不同能耗的能源利用效率 """
        try:
            start_time_str = f"01/01/{year} 00:00:00"
            end_time_str = f"12/31/{year} 23:59:59"
            start_time = transform_timestamp(start_time_str)
            end_time = transform_timestamp(end_time_str)
            async with async_session() as session:
                # 先在 Building 里查 building
                sql_building = select(Building).where(Building.building_id == building_id)
                result_building = await session.execute(sql_building)
                building_data = result_building.scalars().first()
                if not building_data:
                    return {
                        "message": "没有找到建筑信息",
                    }
                building_sqm = building_data.sqm
                # 再在 Energy 里查能耗数据
                sql = select(Energy).where(
                    Energy.building_id == building_id , 
                    Energy.timestamp >= start_time,
                    Energy.timestamp <= end_time
                )
                
                result = await session.execute(sql)
                energy_data = result.scalars().all()
                # 得到全年的四种能耗数据
                all_data_pre_year = add_all_data(energy_data)
                # 计算单位面积能耗
                return {
                    "building_id": building_id,
                    "year": year , 
                    "electricity": all_data_pre_year["electricity"] / building_sqm,
                    "chilledwater": all_data_pre_year["chilledwater"] / building_sqm,
                    "hotwater": all_data_pre_year["hotwater"] / building_sqm,
                    "water": all_data_pre_year["water"] / building_sqm,
                }
        except Exception as e:
            return {"error": "计算单位面积能耗失败", "detail": str(e)}


    @staticmethod
    async def get_building_time_energy_by_hour(
        building_id: str , 
        start_time: datetime,
        end_time: datetime,
    ):

        """ 按照小时的粒度，统计某个建筑在一个时间段内，每小时的能耗数据，方便后续行程曲线等 """
        try:
            async with async_session() as session:
                sql = select(Energy).where(
                    Energy.building_id == building_id,
                    Energy.timestamp >= start_time,
                    Energy.timestamp <= end_time,
                ).order_by(Energy.timestamp.asc())
                result = await session.execute(sql)
                energy_data = result.scalars().all()
                # 桶：整点时刻 -> { meter_type -> 累加值 }（键用完整日期+整点，不能只用 .hour）
                hour_bucket: dict[datetime, defaultdict[str, float]] = defaultdict(
                    lambda: defaultdict(float)
                )
                for e in energy_data:
                    if e.meter_value is None:
                        continue
                    hour_start = _floor_hour(e.timestamp)
                    # 按照小时统计，[小时][能耗类型] = 能耗值
                    hour_bucket[hour_start][e.meter_type] += float(e.meter_value)

                meter_types = ["electricity", "chilledwater", "hotwater", "water"]
                series = []
                for hour_start in sorted(hour_bucket.keys()):
                    meters = hour_bucket[hour_start]
                    row = {"hour": hour_start.isoformat()}
                    for mt in meter_types:
                        row[mt] = meters.get(mt, 0.0)
                    series.append(row)

                return {
                    "building_id": building_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "series": series,
                }
        except Exception as e:
            return {"error": "按照小时的粒度，统计某个建筑在一个时间段内，每小时的能耗数据，方便后续行程曲线等失败", "detail": str(e)}
            

    @staticmethod
    async def get_all_building_id():
        """ 获取所有建筑的 id """
        try:
            building_ids = []
            async with async_session() as session:
                sql = select(Building.building_id)
                result = await session.execute(sql)
                building_ids = result.scalars().all()
                return building_ids
        except Exception as e:
            return {"error": "获取所有建筑的 id 失败", "detail": str(e)}