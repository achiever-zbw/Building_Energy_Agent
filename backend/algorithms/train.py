"""
能耗时序预测 —— 数据预处理 + LSTM 训练入口（修复版）
"""
from __future__ import annotations

import asyncio
import copy
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from db.session import async_session
from models.building import Building
from models.energy import Energy
from models.weather import Weather
from algorithms.models.model import MainModel


# 超参数
BUILDING_ID  = "Moose_education_Lori"
METER_TYPE   = "electricity"
SEQ_LEN      = 168             # 7天历史
HORIZON      = 24              # 预测24小时
HIDDEN_SIZE  = 64              # 适当增加隐层，赋予捕捉峰值的能力
NUM_LAYERS   = 1               # 1层足够
DROPOUT      = 0.1             # 降低Dropout，防止对极值进行平滑
EPOCHS       = 30              # 保持训练轮数
BATCH_SIZE   = 32
LR           = 1e-3
# 按公历切分（与 df['hour'] 对齐，左闭右开）
CALENDAR_YEAR = 2016  # 与数据集年份一致；跨年可改为从 df 推断
TRAIN_START_MONTH, TRAIN_START_DAY = 1, 1
# 训练标签至「不含」此日 0 点 → [1/1, 3/1) 为 1～2 月（约 60 天，闰年 2 月）
TRAIN_END_MONTH, TRAIN_END_DAY = 3, 1
# 验证整月，与训练不重叠 → [4/1, 5/1) 为四月共 30 天（2016）
VAL_START_MONTH, VAL_START_DAY = 3, 2
VAL_END_MONTH, VAL_END_DAY = 3, 8
SAVE_PATH   = Path(__file__).parent / "checkpoints" / "lstm_energy.pth"


async def fetch_data(building_id: str, meter_type: str) -> pd.DataFrame:
    """获取并预处理数据"""
    async with async_session() as session:
        res = await session.execute(
            select(Building).where(Building.building_id == building_id)
        )
        building = res.scalars().first()
        if building is None:
            raise ValueError(f"找不到建筑 {building_id}")
        site_id = building.site_id

        res = await session.execute(
            select(Energy).where(
                Energy.building_id == building_id,
                Energy.meter_type == meter_type,
            ).order_by(Energy.timestamp.asc())
        )
        energy_rows = res.scalars().all()

        res = await session.execute(
            select(Weather).where(
                Weather.site_id == site_id
            ).order_by(Weather.timestamp.asc())
        )
        weather_rows = res.scalars().all()

    if not energy_rows:
        raise ValueError(f"建筑 {building_id} 无 {meter_type} 数据")

    def floor_hour(ts: datetime) -> datetime:
        return ts.replace(minute=0, second=0, microsecond=0)

    # 能耗聚合
    energy_df = pd.DataFrame([
        {"hour": floor_hour(e.timestamp), "energy": e.meter_value or 0.0}
        for e in energy_rows
    ])
    energy_df = energy_df.groupby("hour", as_index=False)["energy"].sum()

    # 气象聚合（修复 NaN 问题）
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
    weather_cols = ["air_temperature", "dew_temperature",
                    "wind_speed", "cloud_coverage", "precipitation"]
    
    if not weather_df.empty:
        weather_df = weather_df.groupby("hour", as_index=False)[weather_cols].mean()

    # 密铺时间轴
    t_min, t_max = energy_df["hour"].min(), energy_df["hour"].max()
    full_hours = pd.DataFrame({"hour": pd.date_range(t_min, t_max, freq="h")})

    merged = full_hours.merge(energy_df, on="hour", how="left")
    # 缺测小时不要用 0（易与「真零」混淆）。库中该小时无行时 left merge 为 NaN，用插值+ffill/bfill 补齐。
    merged["energy"] = (
        merged["energy"]
        .interpolate(method="linear", limit_direction="both")
        .ffill()
        .bfill()
    )
    # 仍无值（整列无数据）时兜底
    merged["energy"] = merged["energy"].fillna(0.0)

    if not weather_df.empty:
        merged = merged.merge(weather_df, on="hour", how="left")
        for col in weather_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0.0)
    else:
        for col in weather_cols:
            merged[col] = 0.0

    merged = merged.sort_values("hour").reset_index(drop=True)
    print(f"[数据] 共 {len(merged)} 小时, 范围 {t_min} ~ {t_max}")
    
    # 数据质量报告
    zero_ratio = (merged["energy"] == 0).mean()
    print(f"[数据] 零能耗占比: {zero_ratio:.2%}")
    
    return merged


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """添加时间特征"""
    hour = df["hour"].dt.hour
    dow = df["hour"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(float)
    return df


def fit_scaler(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """返回 (mean, std)"""
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    std[std < 1e-8] = 1.0
    return mean, std


def transform(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (arr - mean) / std


def inverse_transform(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return arr * std + mean


def window_indices_for_label_range(
    hour_series: pd.Series,
    ts_lo: pd.Timestamp,
    ts_hi_excl: pd.Timestamp,
    seq_len: int,
    horizon: int,
) -> list[int]:
    """窗口 i 合法且未来 horizon 个标签时刻均落在 [ts_lo, ts_hi_excl)。"""
    n = len(hour_series)
    hi = n - seq_len - horizon + 1
    out: list[int] = []
    for i in range(max(0, hi)):
        t0 = hour_series.iloc[i + seq_len]
        t1 = hour_series.iloc[i + seq_len + horizon - 1]
        if t0 >= ts_lo and t1 < ts_hi_excl:
            out.append(i)
    return out


def make_sequences_at_indices(
    features: np.ndarray,
    targets: np.ndarray,
    seq_len: int,
    horizon: int,
    indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    if not indices:
        return (
            np.zeros((0, seq_len, features.shape[1]), dtype=np.float32),
            np.zeros((0, horizon), dtype=np.float32),
        )
    X = np.stack([features[i : i + seq_len] for i in indices]).astype(np.float32)
    y = np.stack([targets[i + seq_len : i + seq_len + horizon] for i in indices]).astype(
        np.float32
    )
    return X, y


async def main():
    # 1. 获取数据
    df = await fetch_data(BUILDING_ID, METER_TYPE)
    df = add_time_features(df)
    
    # 2. 特征工程 - 必须包含能量本身，因为预测范围是 seq_len 之后，特征结束于 seq_len - 1。
    # LSTM 原生处理序列，不需要手动构造 shift 列。
    feature_cols = [
        "energy",  # 核心特征，不包含会导致模型在最后一步“失明”1小时
        "air_temperature", "dew_temperature",
        "wind_speed", "cloud_coverage", "precipitation",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend",
    ]
    target_col = "energy"
    
    # 删除 NaN
    df = df.dropna().reset_index(drop=True)
    print(f"[特征] 使用特征: {feature_cols}")
    print(f"[特征] 特征数量: {len(feature_cols)}")
    
    feat_arr = df[feature_cols].to_numpy(dtype=np.float64)
    target_arr = df[target_col].to_numpy(dtype=np.float64)
    n = len(feat_arr)
    h = df["hour"]

    ts_train_start = pd.Timestamp(
        datetime(CALENDAR_YEAR, TRAIN_START_MONTH, TRAIN_START_DAY, 0, 0, 0)
    )
    ts_train_end = pd.Timestamp(
        datetime(CALENDAR_YEAR, TRAIN_END_MONTH, TRAIN_END_DAY, 0, 0, 0)
    )
    ts_val_start = pd.Timestamp(
        datetime(CALENDAR_YEAR, VAL_START_MONTH, VAL_START_DAY, 0, 0, 0)
    )
    ts_val_end = pd.Timestamp(
        datetime(CALENDAR_YEAR, VAL_END_MONTH, VAL_END_DAY, 0, 0, 0)
    )
    train_span_days = (ts_train_end - ts_train_start).days
    val_span_days = (ts_val_end - ts_val_start).days

    print(
        f"[切分] 训练标签时段 [{ts_train_start}, {ts_train_end})  "
        f"（约 {train_span_days} 天）| "
        f"验证标签时段 [{ts_val_start}, {ts_val_end}) 约 {val_span_days} 天"
    )
    if h.iloc[0] > ts_train_start:
        print(
            f"[提示] 特征表首行时刻为 {h.iloc[0]}，早于该时刻的小时不能作为预测目标（lag+dropna）。"
        )

    train_row_mask = (h >= ts_train_start) & (h < ts_train_end)
    if train_row_mask.sum() == 0:
        raise ValueError(
            f"训练时段内无数据行，请检查 CALENDAR_YEAR 与 CSV 时间是否覆盖 "
            f"{ts_train_start} ~ {ts_train_end}"
        )

    # 特征/目标标准化：仅用「训练时段」内的行 fit，全序列 transform
    feat_mean, feat_std = fit_scaler(feat_arr[train_row_mask.to_numpy()])
    feat_all_s = transform(feat_arr, feat_mean, feat_std)
    target_mean = float(target_arr[train_row_mask.to_numpy()].mean())
    target_std = float(target_arr[train_row_mask.to_numpy()].std())
    if target_std < 1e-8:
        target_std = 1.0
    target_all_s = (target_arr - target_mean) / target_std

    idx_train = window_indices_for_label_range(
        h, ts_train_start, ts_train_end, SEQ_LEN, HORIZON
    )
    idx_val = window_indices_for_label_range(
        h, ts_val_start, ts_val_end, SEQ_LEN, HORIZON
    )
    if not idx_train:
        raise ValueError(
            "训练集窗口数为 0：需整表时间覆盖训练段，且 SEQ_LEN+HORIZON 不超出该段。"
        )
    if not idx_val:
        raise ValueError(
            "验证集窗口数为 0：请确认数据覆盖验证月份且长度足够。"
        )

    X_train, y_train = make_sequences_at_indices(
        feat_all_s, target_all_s, SEQ_LEN, HORIZON, idx_train
    )
    X_test, y_test = make_sequences_at_indices(
        feat_all_s, target_all_s, SEQ_LEN, HORIZON, idx_val
    )

    val_row_mask = (h >= ts_val_start) & (h < ts_val_end)
    ev_raw = df.loc[val_row_mask, "energy"]
    if len(ev_raw) > 0:
        print(
            f"[验证段原始能耗] 时刻 {ts_val_start} ~ {ts_val_end - pd.Timedelta(hours=1)} | "
            f"min={ev_raw.min():.4f} max={ev_raw.max():.4f} mean={ev_raw.mean():.4f} | "
            f"非零小时数 {(ev_raw > 0).sum()}/{len(ev_raw)}"
        )
    else:
        print("[验证段原始能耗] 验证时段内无表行（检查数据是否覆盖验证月份）")

    print(f"[样本] 训练: {len(X_train)}, 验证: {len(X_test)}")
    print(f"[形状] X_train: {X_train.shape}, y_train: {y_train.shape}")
    
    # 6. DataLoader：训练集 shuffle 可避免按时间顺序批次的优化偏置，不造成泄露（每行已是独立窗口）
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 7. 模型
    input_size = len(feature_cols)
    model = MainModel(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        output_size=HORIZON,
        dropout_rate=DROPOUT
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-6) # 减弱L2正则，允许权重拟合尖峰
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.L1Loss() # 使用L1Loss（MAE），通常能比MSE给出更锐利的预测边界
    
    # 8. 训练（带早停）
    # 注意：state_dict().copy() 只拷贝 dict，张量仍与参数共享存储，必须用深拷贝保存最佳权重。
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state: dict[str, torch.Tensor] | None = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 梯度裁剪
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_dl:
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
        val_loss /= len(test_ds)
        
        scheduler.step(val_loss)
        
        # 早停
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print(f"Early stopping at epoch {epoch}")
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

        if epoch % 1 == 0:
            print(f"Epoch {epoch:3d}/{EPOCHS}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")
    else:
        # 未触发早停跑满 EPOCHS：仍应使用验证集最优权重，而非最后一轮
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

    # 9. 保存模型
    SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "feat_mean": feat_mean,
        "feat_std": feat_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "feature_cols": feature_cols,
        "seq_len": SEQ_LEN,
        "horizon": HORIZON,
    }, SAVE_PATH)
    print(f"[保存] 模型已保存至 {SAVE_PATH}")
    
    # 10. 评估
    evaluate(model, test_dl, y_test, target_mean, target_std, HORIZON)


def evaluate(model, test_dl, y_test_raw, target_mean, target_std, horizon):
    """评估模型"""
    model.eval()
    preds_list, trues_list = [], []
    
    with torch.no_grad():
        for xb, yb in test_dl:
            pred = model(xb)
            preds_list.append(pred.cpu().numpy())
            trues_list.append(yb.cpu().numpy())
    
    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)
    
    preds_real = preds * target_std + target_mean
    trues_real = trues * target_std + target_mean
    
    # 计算指标
    mae = np.mean(np.abs(preds_real - trues_real))
    rmse = np.sqrt(np.mean((preds_real - trues_real) ** 2))
    
    # 安全的 MAPE（只计算真实值 > 0.1 的点）
    mask = trues_real > 0.1
    if mask.any():
        mape = np.mean(np.abs((trues_real[mask] - preds_real[mask]) / trues_real[mask])) * 100
    else:
        mape = np.nan
    
    # 添加 R²
    ss_res = np.sum((trues_real - preds_real) ** 2)
    ss_tot = np.sum((trues_real - np.mean(trues_real)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    print("\n" + "=" * 50)
    print(f"  验证集评估结果（{len(preds)} 个样本，预测 {horizon} 小时）")
    print(f"  MAE  : {mae:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAPE : {mape:.2f}%" if not np.isnan(mape) else "  MAPE : N/A")
    print(f"  R²   : {r2:.4f}")
    print("=" * 50)
    
    # 绘制预测 vs 真实图
    plot_results(preds_real[:, 0], trues_real[:, 0], mae, rmse, mape)


def plot_results(predictions, actuals, mae, rmse, mape):
    """绘制预测结果"""
    n = min(336, len(predictions))  # 最多显示14天
    pred_seq = predictions[-n:]
    true_seq = actuals[-n:]
    x = np.arange(n)
    
    plt.figure(figsize=(14, 6))
    plt.plot(x, true_seq, label='Actual', linewidth=1.5, alpha=0.8)
    plt.plot(x, pred_seq, label='Predicted', linewidth=1.5, linestyle='--', alpha=0.8)
    plt.fill_between(x, true_seq, pred_seq, alpha=0.1)
    plt.xlabel('Time Step (Hour)')
    plt.ylabel('Energy')
    plt.title(f'Predicted vs Actual\nMAE={mae:.3f}, RMSE={rmse:.3f}, MAPE={mape:.2f}%')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = Path(__file__).parent / "checkpoints" / "forecast_result.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[图表] 保存至 {plot_path}")


if __name__ == "__main__":
    asyncio.run(main())