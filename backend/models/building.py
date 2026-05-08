from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean
from db.base import Base

class Building(Base):
    """
    建筑信息表
    """
    __tablename__ = "buildings"
    # 主键自增
    id = Column(Integer, primary_key=True)
    # 建筑编号，唯一标识
    building_id = Column(String(255), index=True, unique=True, nullable=False)
    # 所属园区
    site_id = Column(String(255), index=True, nullable=False)
    # 建筑用途
    primaryspaceusage = Column(String(255), nullable=False)
    # 建筑面积
    sqm = Column(Float, nullable=False)
    # 人数
    occupants = Column(Integer, nullable=False)
    # 建成年份
    yearbuilt = Column(Integer, nullable=False)
    # 是否安装电表
    have_electricity = Column(Boolean, nullable=False)
    # 是否安装冷冻水表
    have_chilledwater = Column(Boolean, nullable=False)
    # 是否安装热水表
    have_hotwater = Column(Boolean, nullable=False)
    # 是否安装水表
    have_water = Column(Boolean, nullable=False)