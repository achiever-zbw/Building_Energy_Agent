from sqlalchemy import Column, Integer, String, DateTime, Float, Boolean, UniqueConstraint
from db.base import Base

class Weather(Base):
    __tablename__ = "weather"

    __table_args__ = (
        UniqueConstraint("site_id", "timestamp", name="uq_weather_site_time"),
    )

    id = Column(Integer, primary_key=True)
    site_id = Column(String(255), nullable=False)
    # 气温
    air_temperature = Column(Float, nullable=True)
    # 露点温度
    dew_temperature = Column(Float, nullable=True)
    # 风速
    wind_speed = Column(Float, nullable=True)
    # 云量
    cloud_coverage = Column(Float, nullable=True)
    # 降水量
    precipitation = Column(Float, nullable=True)
    timestamp = Column(DateTime, nullable=True)