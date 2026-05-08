from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, ForeignKey, UniqueConstraint, Index
from db.base import Base

class Energy(Base):
    """能耗数据表"""
    __tablename__ = "energy"
    __table_args__ = (
        # 同一栋楼、同一时刻、同一表计类型只能有一条记录
        UniqueConstraint(
            "building_id",
            "timestamp",
            "meter_type",
            name="uq_energy_building_time_meter",
        ),
        # 联合索引，查询
        Index("idx_energy_building_time" , "building_id", "timestamp"), 
        Index("idx_energy_building_time_meter" , "building_id", "timestamp", "meter_type"),
    )

    # 自增主键（便于 ORM 引用、批量操作）
    id = Column(Integer, primary_key=True, autoincrement=True)

    # 关联 building_id
    building_id = Column(String(255), ForeignKey("buildings.building_id"), nullable=False)
    # 时间戳
    timestamp = Column(DateTime, nullable=False)
    # 能耗种类
    meter_type = Column(String(255), nullable=False)
    # 能耗数值
    meter_value = Column(Float)