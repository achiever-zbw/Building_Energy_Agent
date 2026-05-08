"""
能耗预测服务
"""
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import async_session
from models.building import Building
from models.energy import Energy
from models.weather import Weather
from algorithms.models.model import MainModel

class ForecastService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ForecastService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
            
        # 记录模型文件路径
        self.checkpoint_path = Path(__file__).resolve().parent.parent / "algorithms" / "checkpoints" / "lstm_energy.pth"
        self.model = None
        self.feat_mean = None
        self.feat_std = None
        self.target_mean = None
        self.target_std = None
        self.feature_cols = None
        self.seq_len = 168
        self.horizon = 24
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialized = True

    def load_model(self):
        """单例加载模型参数与预处理字典"""
        if self.model is not None:
            return

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"找不到模型权重文件: {self.checkpoint_path}")

        # weights_only=False 允许加载包含 numpy 数组的字典（我们在保存模型时存了 numpy array 类型的 feat_mean 等）
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        self.feat_mean = checkpoint["feat_mean"]
        self.feat_std = checkpoint["feat_std"]
        self.target_mean = checkpoint["target_mean"]
        self.target_std = checkpoint["target_std"]
        self.feature_cols = checkpoint["feature_cols"]
        self.seq_len = checkpoint.get("seq_len", 168)
        self.horizon = checkpoint.get("horizon", 24)

        input_size = len(self.feature_cols)
        # 与 train.py 里最新架构保持一致 (隐藏层大小为 64)
        self.model = MainModel(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            output_size=self.horizon,
            dropout_rate=0.1
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device)
        self.model.eval()

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加时间特征（和训练时完全一致）"""
        hour = df["hour"].dt.hour
        dow = df["hour"].dt.dayofweek
        df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
        df["is_weekend"] = (dow >= 5).astype(float)
        return df

    async def fetch_historical_data(self, building_id: str, meter_type: str, target_date: str) -> pd.DataFrame:
        """
        从数据库中查询 target_date 之前的 168 小时能耗与气象数据
        """
        target_ts = pd.Timestamp(target_date)
        start_ts = target_ts - pd.Timedelta(hours=self.seq_len)
        
        async with async_session() as session:
            # 查询 building
            res = await session.execute(
                select(Building).where(Building.building_id == building_id)
            )
            building = res.scalars().first()
            if building is None:
                raise ValueError(f"找不到建筑 {building_id}")
            
            site_id = building.site_id

            # 查询能耗
            res = await session.execute(
                select(Energy).where(
                    Energy.building_id == building_id,
                    Energy.meter_type == meter_type,
                    Energy.timestamp >= start_ts,
                    Energy.timestamp < target_ts
                ).order_by(Energy.timestamp.asc())
            )
            energy_rows = res.scalars().all()

            # 查询气象
            res = await session.execute(
                select(Weather).where(
                    Weather.site_id == site_id,
                    Weather.timestamp >= start_ts,
                    Weather.timestamp < target_ts
                ).order_by(Weather.timestamp.asc())
            )
            weather_rows = res.scalars().all()

        def floor_hour(ts: datetime) -> datetime:
            return ts.replace(minute=0, second=0, microsecond=0)

        # 聚合能耗
        energy_df = pd.DataFrame([
            {"hour": floor_hour(e.timestamp), "energy": e.meter_value or 0.0}
            for e in energy_rows
        ])
        if energy_df.empty:
            raise ValueError(f"过去7天 ({start_ts} 到 {target_ts}) 没有任何能耗数据，无法预测。")
            
        energy_df = energy_df.groupby("hour", as_index=False)["energy"].sum()

        # 聚合气象
        weather_records = []
        for w in weather_rows:
            precip = w.precipitation
            if precip is None or (isinstance(precip, float) and math.isnan(precip)):
                precip = 0.0
            
            weather_records.append({
                "hour": floor_hour(w.timestamp),
                "air_temperature": w.air_temperature or 0.0,
                "dew_temperature": w.dew_temperature or 0.0,
                "wind_speed": w.wind_speed or 0.0,
                "cloud_coverage": w.cloud_coverage or 0.0,
                "precipitation": precip,
            })
        
        weather_df = pd.DataFrame(weather_records)
        weather_cols = ["air_temperature", "dew_temperature", "wind_speed", "cloud_coverage", "precipitation"]
        
        if not weather_df.empty:
            weather_df = weather_df.groupby("hour", as_index=False)[weather_cols].mean()

        # 密铺这168个小时
        full_hours = pd.DataFrame({"hour": pd.date_range(start_ts, target_ts - pd.Timedelta(hours=1), freq="h")})
        merged = full_hours.merge(energy_df, on="hour", how="left")
        
        # 填充能耗
        merged["energy"] = merged["energy"].interpolate(method="linear", limit_direction="both").ffill().bfill().fillna(0.0)

        # 填充气象
        if not weather_df.empty:
            merged = merged.merge(weather_df, on="hour", how="left")
            for col in weather_cols:
                if col in merged.columns:
                    merged[col] = merged[col].interpolate(method="linear", limit_direction="both").ffill().bfill().fillna(0.0)
        else:
            for col in weather_cols:
                merged[col] = 0.0

        merged = merged.sort_values("hour").reset_index(drop=True)
        return merged

    async def predict(self, building_id: str, meter_type: str, target_date: str) -> list[dict]:
        """
        进行预测主逻辑
        """
        self.load_model()

        # 获取数据 (168 行)
        df = await self.fetch_historical_data(building_id, meter_type, target_date)
        
        if len(df) != self.seq_len:
            raise ValueError(f"数据缺失严重，预期的历史小时数为 {self.seq_len}，但构造后为 {len(df)}")

        # 添加时间特征
        df = self.add_time_features(df)

        # 提取模型所需特征
        feat_arr = df[self.feature_cols].to_numpy(dtype=np.float32)
        
        # 归一化 (使用训练时保存的 mean, std)
        feat_arr_scaled = (feat_arr - self.feat_mean) / self.feat_std
        
        # 变张量 (1, 168, 11) - 强制转为 float32 以匹配模型权重类型
        x_tensor = torch.from_numpy(feat_arr_scaled).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            pred_scaled = self.model(x_tensor) # (1, 24)
            
        pred_scaled_np = pred_scaled.cpu().numpy()[0]
        
        # 反归一化
        pred_real = pred_scaled_np * self.target_std + self.target_mean
        # 防止出现负数预测（能耗不能为负）
        pred_real = np.maximum(pred_real, 0.0)
        
        # 组装返回结果
        target_ts = pd.Timestamp(target_date)
        forecast_times = pd.date_range(target_ts, target_ts + pd.Timedelta(hours=self.horizon - 1), freq="h")
        
        results = []
        for i in range(self.horizon):
            results.append({
                "timestamp": forecast_times[i].strftime("%Y-%m-%d %H:%M:%S"),
                "predicted_energy": round(float(pred_real[i]), 4)
            })
            
        return results

# 全局单例
forecast_service = ForecastService()
