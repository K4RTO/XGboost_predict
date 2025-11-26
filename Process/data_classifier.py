import os, sys, gc
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR   = Path(__file__).resolve().parents[1]
DATA_DIR   = BASE_DIR / "Data"
INPUT_FILE = DATA_DIR / "origin_data.xlsx"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 配置要切的日期窗口（含首尾）
START_DATE = "2025-08-20"
END_DATE   = "2025-08-23"

def parse_mixed_datetime(series: pd.Series) -> pd.Series:
    s = series.copy()

    s_str = s.astype(str).str.strip().str.replace(r"[./]", "-", regex=True)
    dt_str = pd.to_datetime(s_str, errors="coerce")

    s_num = pd.to_numeric(s, errors="coerce")
    dt_num = pd.to_datetime(s_num, unit="D", origin="1899-12-30", errors="coerce")
    return dt_str.fillna(dt_num)

def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"未找到原始数据：{INPUT_FILE}")

    df = pd.read_excel(INPUT_FILE)

    if "班次" not in df.columns:
        raise ValueError("源文件缺少 '班次' 列")

    # 解析日期并筛选窗口
    dt = parse_mixed_datetime(df["班次"])
    df["_dt"] = dt
    df["_date"] = dt.dt.date

    start_date = pd.to_datetime(START_DATE).date()
    end_date   = pd.to_datetime(END_DATE).date()
    mask = (df["_date"] >= start_date) & (df["_date"] <= end_date)

    current = df.loc[mask].copy()
    remain  = df.loc[~mask].copy()

    # 给当期样本稳定行号（后续全程按照 ROW_ID 一一对应）
    current.insert(0, "ROW_ID", np.arange(1, len(current)+1, dtype=np.int64))

    # 导出 call_off（去订单类型）/ truth（保留订单类型）
    call_off = current.drop(columns=["订单类型"], errors="ignore")
    call_off_out = DATA_DIR / "call_off.xlsx"
    call_off.to_excel(call_off_out, index=False)
    print(f"[导出] call_off.xlsx -> {call_off_out} （{len(call_off)} 行）")

    truth_out = DATA_DIR / "truth.xlsx"
    current.to_excel(truth_out, index=False)
    print(f"[导出] truth.xlsx    -> {truth_out} （{len(current)} 行）")

    # 训练原始数据
    data_out = DATA_DIR / "data.xlsx"
    remain.to_excel(data_out, index=False)
    print(f"[导出] data.xlsx     -> {data_out} （{len(remain)} 行）")

if __name__ == "__main__":
    main()
    gc.collect()
    sys.exit(0)