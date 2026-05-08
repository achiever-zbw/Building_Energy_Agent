"""
能耗异常检测 —— 核心算法层（无数据库、无 HTTP）。
业务层职责：从库中按建筑/表计/时间范围查询，按小时聚合为有序数值序列，
再调用本模块的 detect_rolling_zscore。
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Literal, Sequence

# 异常类型 : spike: 异常高, drop: 异常低
AnomalyKind = Literal["spike", "drop"]

@dataclass(frozen=True)
class ZScoreAnomalyPoint:
    """单个异常点（对应原序列中的下标）。"""
    index: int              # 异常点在原序列的位置
    value: float            # 异常点的值
    baseline_mean: float    # 基准窗口的均值
    baseline_std: float     # 基准窗口的标准差
    z_score: float          # 计算出的 z-score 值
    kind: AnomalyKind       # 异常类型


@dataclass
class ZScoreAnomalyResult:
    """ 一次检测的完整结果 """

    anomalies: list[ZScoreAnomalyPoint] = field(default_factory=list)
    """判定为异常的点（仅包含 evaluation 范围内的下标）。"""

    skipped_insufficient_baseline: list[int] = field(default_factory=list)
    """当前点之前窗口内有效数据不足 min_points，未参与判定。"""

    skipped_flat_baseline: list[int] = field(default_factory=list)
    """基准标准差过小（≈常数序列），未做 Z-score 判定。"""

    skipped_non_finite: list[int] = field(default_factory=list)
    """当前点值为 nan/inf，跳过。"""


def _finite(xs: Sequence[float]) -> list[float]:
    return [x for x in xs if math.isfinite(x)]


def _mean_std(sample: list[float]) -> tuple[float, float] | None:
    """样本均值与样本标准差；点数 < 2 返回 None。"""
    n = len(sample)
    if n < 2:
        return None
    m = sum(sample) / n
    var = sum((x - m) ** 2 for x in sample) / (n - 1)
    if var < 0:
        var = 0.0
    return m, math.sqrt(var)



def detect_rolling_zscore(
    values: Sequence[float],
    *,
    window: int,
    z_threshold: float,
    min_points: int = 2,
    eps: float = 1e-9,
    detect_drop: bool = True,
    evaluation_start: int | None = None,
    evaluation_end_exclusive: int | None = None,
) -> ZScoreAnomalyResult:
    """
    滚动窗口 Z-score 异常检测。
    对每个下标 i（从 window 到 len-1）：
        - 基准集为 values[i-window : i] 中的有限数值；
        - 若基准点数 < min_points 或 std < eps，则不判定；
        - 否则 z = (values[i] - mean) / max(std, eps)；
        - 若 z > z_threshold 为 spike；若 detect_drop 且 z < -z_threshold 为 drop。
    仅当下标 i 落在 [evaluation_start, evaluation_end_exclusive) 内时，
    才把异常写入 anomalies（便于「整段序列含历史，但只报某个月」）。
    若 evaluation_start / evaluation_end_exclusive 为 None，则默认为
    start=window, end=len(values)。
    参数y
    ----
    values
        按时间升序的一维序列（如按小时聚合后的用电量），与业务层约定一致即可。
    window: 窗口大小
        严格早于当前点的样本个数（不是日历小时数；业务层应传入「前 W 个点」）。
    z_threshold: 异常阈值，超过阈值判定为异常
        阈值，常用 2.0 ~ 3.0。
    min_points: 基准集中至少多少个有限样本才计算 Z-score（至少为 2 才有样本标准差）。
        基准集中至少多少个有限样本才计算 Z-score（至少为 2 才有样本标准差）。
    eps: 防止 std 过小导致除法爆炸；小于 eps 的 std 视为「平坦基准」并跳过。
    detect_drop: 是否检测异常偏低（z < -threshold）。
    evaluation_start: 只在此下标范围内输出异常；默认从 window 起到序列末尾。
    evaluation_end_exclusive: 只在此下标范围内输出异常；默认从 window 起到序列末尾。
    """
    # 创建结果对象
    result = ZScoreAnomalyResult()

    # 获取序列长度
    n = len(values)
    if window < 1 or n <= window:
        return result
    
    ev_start = window if evaluation_start is None else evaluation_start
    ev_end = n if evaluation_end_exclusive is None else evaluation_end_exclusive

    for i in range(window, n):
        cur = values[i]
        # 过滤非法值
        if not math.isfinite(cur):
            if ev_start <= i < ev_end:
                result.skipped_non_finite.append(i)
            continue
        baseline = _finite(values[i - window : i])

        if len(baseline) < min_points:
            if ev_start <= i < ev_end:
                result.skipped_insufficient_baseline.append(i)
            continue

        # 计算均值和标准差
        ms = _mean_std(baseline)
        if ms is None:
            if ev_start <= i < ev_end:
                result.skipped_insufficient_baseline.append(i)
            continue

    
        mu, sigma = ms
        if sigma < eps:
            if ev_start <= i < ev_end:
                result.skipped_flat_baseline.append(i)
            continue
        # 计算 z-score
        z = (cur - mu) / max(sigma, eps)
        kind: AnomalyKind | None = None
        # 异常升高
        if z > z_threshold:
            kind = "spike"
        # 异常降低
        elif detect_drop and z < -z_threshold:
            kind = "drop"
        if kind is not None and ev_start <= i < ev_end :
            result.anomalies.append(
                ZScoreAnomalyPoint(
                    index=i,
                    value=cur,
                    baseline_mean=mu,
                    baseline_std=sigma,
                    z_score=z,
                    kind=kind,
                )
            )
    return result
