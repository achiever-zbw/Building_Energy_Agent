import math
from calendar import monthrange
from collections import defaultdict
from datetime import datetime, timedelta
from sqlalchemy import select
from db.session import async_session
from models.energy import Energy
from algorithms.anomaly import detect_rolling_zscore

# 向下取整到小时
def _floor_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)

# 获取某年某月的起始和结束时间
def _month_bounds(year: int, month: int) -> tuple[datetime, datetime]:
    start = datetime(year, month, 1, 0, 0, 0)
    last_day = monthrange(year, month)[1]
    end = datetime(year, month, last_day, 23, 0, 0)
    return start, end

class AnomalyService:
    """ 异常检测服务 """
    @staticmethod
    async def analyze_building_energy_month_zscore(
        building_id: str,
        meter_type: str,
        year: int,
        month: int,
        *,
        window: int = 168,
        z_threshold: float = 3.5,
        min_points: int = 48,
        detect_drop: bool = True,
    ) -> dict:
        """
        业务入口：某建筑、某表计类型、某年某月是否出现异常（Z-score）。

        做法：从「月初前留出 window 小时」起拉到月末，按整点小时聚合 sum(meter_value)，
        再对「落在该月内的小时」做滚动 Z-score 检测。
        """
        month_start, month_end = _month_bounds(year, month)
        query_start = _floor_hour(month_start - timedelta(hours=window))

        async with async_session() as session:
            q = (
                select(Energy)
                .where(
                    Energy.building_id == building_id,
                    Energy.meter_type == meter_type,
                    Energy.timestamp >= query_start,
                    Energy.timestamp <= month_end,
                )
                .order_by(Energy.timestamp.asc())
            )
            result = await session.execute(q)
            rows = result.scalars().all()

        # 按整点聚合
        hour_sums: dict[datetime, float] = defaultdict(float)
        for e in rows:
            if e.meter_value is None:
                continue
            v = float(e.meter_value)
            if not math.isfinite(v):
                continue
            hour_sums[_floor_hour(e.timestamp)] += v

        # 密铺时间轴（缺小时记 0；若你希望「缺测不算 0」可改成 math.nan 并自行调整规则）
        hours: list[datetime] = []
        h = query_start
        end_h = _floor_hour(month_end)
        while h <= end_h:
            hours.append(h)
            h += timedelta(hours=1)

        values = [hour_sums.get(t, 0.0) for t in hours]

        # 仅在该月范围内输出异常：对应下标区间
        eval_start = next((i for i, t in enumerate(hours) if t >= month_start), len(values))
        eval_end = next((i for i, t in enumerate(hours) if t > month_end), len(values))

        algo = detect_rolling_zscore(
            values,
            window=window,
            z_threshold=z_threshold,
            min_points=min_points,
            detect_drop=detect_drop,
            evaluation_start=eval_start,
            evaluation_end_exclusive=eval_end,
        )

        anomalies_out = []
        for a in algo.anomalies:
            anomalies_out.append(
                {
                    "hour": hours[a.index].isoformat(sep=" "),
                    "value": a.value,
                    "baseline_mean": a.baseline_mean,
                    "baseline_std": a.baseline_std,
                    "z_score": a.z_score,
                    "kind": a.kind,
                }
            )

        return {
            "building_id": building_id,
            "meter_type": meter_type,
            "year": year,
            "month": month,
            "query_range": {
                "start": query_start.isoformat(sep=" "),
                "end": month_end.isoformat(sep=" "),
            },
            "params": {
                "window": window,
                "z_threshold": z_threshold,
                "min_points": min_points,
                "detect_drop": detect_drop,
            },
            "summary": {
                "has_anomaly": len(anomalies_out) > 0,
                "anomaly_count": len(anomalies_out),
            },
            "anomalies": anomalies_out,
            "diagnostics": {
                "skipped_insufficient_baseline": len(algo.skipped_insufficient_baseline),
                "skipped_flat_baseline": len(algo.skipped_flat_baseline),
                "skipped_non_finite": len(algo.skipped_non_finite),
            },
        }