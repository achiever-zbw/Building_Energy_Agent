"""
冷水机组 COP 模拟数据服务。

在尚未接入真实冷机流量计与电表点位时，由服务端按建筑编号与时间范围
生成可复现的模拟运行数据（COP = 移除冷量 Q / 机组输入功率 P）。
"""
from __future__ import annotations

import hashlib
import random
from datetime import date, timedelta
from typing import Any


class CopSimulationService:
    """生成模拟 COP 日序列与汇总指标。"""

    TARGET_COP = 5.0

    @classmethod
    def simulate(
        cls,
        building_id: str,
        days: int,
        end_date: date | None = None,
    ) -> dict[str, Any]:
        if days < 1:
            days = 1
        if days > 90:
            days = 90

        end = end_date or date.today()
        if days > 1:
            start = end - timedelta(days=days - 1)
        else:
            start = end

        # 同一建筑编号在同一参数下序列稳定（便于演示复现）
        seed_bytes = f"{building_id}:{start.isoformat()}:{end.isoformat()}:{days}".encode()
        seed = int(hashlib.sha256(seed_bytes).hexdigest()[:12], 16) % (2**31)
        rng = random.Random(seed)

        daily: list[dict[str, Any]] = []
        for i in range(days):
            d = start + timedelta(days=i)
            # 模拟冷负荷与能效波动
            q_kw = 1050.0 + rng.uniform(0, 320.0)
            cop = 4.75 + rng.uniform(0, 0.65)
            p_kw = q_kw / cop
            cop_rounded = round(q_kw / p_kw, 3)

            daily.append(
                {
                    "date": d.isoformat(),
                    "day": d.strftime("%m-%d"),
                    "q_kw": round(q_kw, 1),
                    "p_kw": round(p_kw, 1),
                    "cop": cop_rounded,
                }
            )

        cops = [row["cop"] for row in daily]
        avg_cop = round(sum(cops) / len(cops), 3) if cops else 0.0
        compliant = sum(1 for c in cops if c >= cls.TARGET_COP)
        compliance_rate = round(compliant / len(cops), 4) if cops else 0.0

        return {
            "building_id": building_id,
            "chiller_label": "离心式冷水机组（模拟工况）",
            "target_cop": cls.TARGET_COP,
            "avg_cop": avg_cop,
            "compliance_rate": compliance_rate,
            "compliant_days": compliant,
            "total_days": len(daily),
            "period_start": start.isoformat(),
            "period_end": end.isoformat(),
            "formula": "COP = 移除冷量 Q（kW）÷ 机组输入功率 P（kW）",
            "data_source": "simulation",
            "daily": daily,
        }
