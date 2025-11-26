import gc
import sys
from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_EXCEL = BASE_DIR / 'Data' / 'data.xlsx'          # ← 上一步切片后留下的“用于训练/建模”的历史数据
OUTPUT_EXCEL = BASE_DIR / 'Data' / 'Processed_data.xlsx'
DAY_DATA = BASE_DIR / 'Data' / 'day_data.xlsx'
NIGHT_DATA = BASE_DIR / 'Data' / 'night_data.xlsx'

SHEET_NAME = 0

# 训练会用到的主要字段
FEATURES = ["Line", "Line代码", "Lot", "Model", "model2", "PN", "供应商", "品名", "品类"]

# 保留列（只做导出整洁，训练时你读取 day/night 表）
KEEP_COLS = [
    "班次", "白夜班", "PickTime", "厂别", "库别", "订单类型",
    "Line", "Line代码", "Lot", "Model", "model2", "PN", "供应商", "品名", "品类",
    "PCS数", "托规", "MPQ"
]

def parse_mixed_datetime(series: pd.Series) -> pd.Series:
    """兼容字符串/Excel 序列的日期时间解析"""
    s = series.copy()
    # 字符串路径
    s_str = s.astype(str).str.strip().str.replace(r"[./]", "-", regex=True)
    dt_str = pd.to_datetime(s_str, errors="coerce")
    # Excel 序列路径
    s_num = pd.to_numeric(s, errors="coerce")
    dt_num = pd.to_datetime(s_num, unit="D", origin="1899-12-30", errors="coerce")
    return dt_str.fillna(dt_num)

def coalesce_duplicate_columns(df: pd.DataFrame, cols_to_fix):
    """
    合并同名列：如果同名列被读成多个（例如 'Model', 'Model.1'），
    将这些列横向从左到右 bfill，保留第一个非空值。
    """
    out = df.copy()
    for col in cols_to_fix:
        same = [c for c in out.columns if c == col]
        if len(same) > 1:
            tmp = out.loc[:, same]
            fused = tmp.bfill(axis=1).iloc[:, 0]
            out.drop(columns=same, inplace=True)
            out[col] = fused
    return out

def main():
    if not INPUT_EXCEL.exists():
        raise FileNotFoundError(f"未找到训练原始数据：{INPUT_EXCEL}")

    df = pd.read_excel(INPUT_EXCEL, sheet_name=SHEET_NAME)


    df = coalesce_duplicate_columns(df, FEATURES)

    # 只保留有有效标签的样本
    if "订单类型" not in df.columns:
        raise ValueError("训练数据缺少 '订单类型' 列")
    lbl = df["订单类型"].astype(str).str.strip()
    df = df[lbl.isin(["开线", "接力"])].copy()

    # 解析“班次”并映射白/夜
    if "班次" not in df.columns:
        raise ValueError("训练数据缺少 '班次' 列")
    times = parse_mixed_datetime(df["班次"])
    hours = times.dt.hour
    df["班次"] = np.where(hours == 8, "白班",
                          np.where(hours == 20, "夜班", "未知班次"))
    # 保持 KEEP_COLS 完整性：白夜班 = 班次
    df["白夜班"] = df["班次"]

    # 仅保留所需列（不存在的列会被自动补齐为 NaN）
    for c in KEEP_COLS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[KEEP_COLS]

    # 导出总的 processed
    Path(OUTPUT_EXCEL).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(OUTPUT_EXCEL, index=False)

    # 拆出白/夜班（训练脚本读取这两个文件）
    day_df = df[df["班次"] == "白班"].copy()
    night_df = df[df["班次"] == "夜班"].copy()
    day_df.to_excel(DAY_DATA, index=False)
    night_df.to_excel(NIGHT_DATA, index=False)

    print(f"已生成总表: {OUTPUT_EXCEL}  行数: {len(df)}; 列数: {len(df.columns)}")
    print(f"已生成白班: {DAY_DATA}     行数: {len(day_df)}; 列数: {len(day_df.columns)}")
    print(f"已生成夜班: {NIGHT_DATA}   行数: {len(night_df)}; 列数: {len(night_df.columns)}")

if __name__ == "__main__":
    main()
    gc.collect()
    sys.exit(0)