import os, sys, gc
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

BASE_DIR   = Path(__file__).resolve().parents[1]
DATA_DIR   = BASE_DIR / "Data"
RESULT_DIR = BASE_DIR / "Result"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

TRUTH_FILE        = DATA_DIR / "truth.xlsx"
TRUTH_DAY_OUT     = RESULT_DIR / "truth_day.xlsx"
TRUTH_NIGHT_OUT   = RESULT_DIR / "truth_night.xlsx"

PRED_DAY_FILE     = RESULT_DIR / "predict_day.xlsx"
PRED_NIGHT_FILE   = RESULT_DIR / "predict_night.xlsx"

COMPARE_DAY_OUT   = RESULT_DIR / "compare_day.xlsx"
COMPARE_NIGHT_OUT = RESULT_DIR / "compare_night.xlsx"
SUMMARY_TXT       = RESULT_DIR / "summary.txt"

# —— 严格按这 9 个字段做键（再配合 __occ 逐条对齐）——
KEY_COLS = ["Line","Line代码","Lot","Model","model2","PN","供应商","品名","品类"]


KEEP_COLS = ["班次","PickTime","厂别","库别","订单类型",
             "Line","Line代码","Lot","Model","model2","PN","供应商","品名","品类",
             "PCS数","托规","MPQ","prob_开线","原始订单类型","__occ","命中"]

def log(msg: str):
    print(msg)
    with open(SUMMARY_TXT, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def parse_shift_to_daynight(series: pd.Series) -> pd.Series:
    """把“班次”解析为时间，再映射为‘白班’(8点) / ‘夜班’(20点) / 其他→‘未知班次’"""
    s = series.copy()

    # 数字（Excel 序列）路径
    mask_num = pd.to_numeric(s, errors="coerce").notna()
    dt_num = pd.to_datetime(pd.to_numeric(s[mask_num], errors="coerce"),
                            unit="D", origin="1899-12-30", errors="coerce")

    # 字符串路径：统一分隔符，再解析
    s_str = s[~mask_num].astype(str).str.strip()
    s_norm = s_str.str.replace(r"[./]", "-", regex=True)
    dt_str = pd.to_datetime(s_norm, errors="coerce")

    # 合并
    dt = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    dt.loc[mask_num] = dt_num
    dt.loc[~mask_num] = dt_str

    hour = dt.dt.hour
    return np.where(hour == 8, "白班", np.where(hour == 20, "夜班", "未知班次"))

def _prep_for_join(df: pd.DataFrame) -> pd.DataFrame:
    """对齐前统一：数值列转数值，其余转字符串去空格。"""
    out = df.copy()
    for c in out.columns:
        if c in ["PCS数","托规","MPQ"]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = out[c].astype(str).str.strip()
    return out

def _attach_occ(df: pd.DataFrame) -> pd.DataFrame:
    """在 KEY_COLS 上对每条出现做分身编号 __occ（从 0 开始），保证逐条一一配对。"""
    if not set(KEY_COLS).issubset(df.columns):
        # 不足键列时也给个 __occ=0，避免后续报错
        out = df.copy()
        out["__occ"] = 0
        return out
    out = df.copy()
    out["__occ"] = out.groupby(KEY_COLS).cumcount()
    return out

def read_truth_and_split():
    if not TRUTH_FILE.exists():
        raise FileNotFoundError(f"未找到 truth 文件：{TRUTH_FILE}")
    df = pd.read_excel(TRUTH_FILE)

    # 只保留有标签（开线/接力）的记录 —— 基数以此为准
    lbl = df["订单类型"].astype(str).str.strip()
    df = df[lbl.isin(["开线","接力"])].copy()

    # 生成白/夜班文本
    dn = parse_shift_to_daynight(df["班次"])
    df["班次"] = dn

    # 导出（列顺序友好，不影响对齐）
    keep = [c for c in KEEP_COLS if c in df.columns]
    df = df[keep].copy()

    truth_day   = df[df["班次"] == "白班"].copy()
    truth_night = df[df["班次"] == "夜班"].copy()

    # 为后续逐条对齐准备 __occ
    truth_day   = _attach_occ(truth_day)
    truth_night = _attach_occ(truth_night)

    truth_day.to_excel(TRUTH_DAY_OUT, index=False)
    truth_night.to_excel(TRUTH_NIGHT_OUT, index=False)

    log(f"[导出] truth_day   -> {TRUTH_DAY_OUT} （{len(truth_day)} 行）")
    log(f"[导出] truth_night -> {TRUTH_NIGHT_OUT} （{len(truth_night)} 行）")

def _one_to_one_compare(truth_path: Path, pred_path: Path, compare_out: Path, title: str):
    if not truth_path.exists():
        log(f"[跳过] 未找到 truth：{truth_path}")
        return
    if not pred_path.exists():
        log(f"[跳过] 未找到预测：{pred_path}")
        return

    # 读入
    truth_raw = pd.read_excel(truth_path)
    pred_raw  = pd.read_excel(pred_path)

    # 清洗
    truth = _prep_for_join(truth_raw)
    pred  = _prep_for_join(pred_raw)

    # 只保留双方都有的键列
    use_keys = [k for k in KEY_COLS if k in truth.columns and k in pred.columns]
    if len(use_keys) != len(KEY_COLS):
        miss = set(KEY_COLS) - set(use_keys)
        log(f"[警告] 缺少对齐键: {sorted(list(miss))}，将用可用键对齐。")

    # 分身编号，保证“逐条对齐”（即便键相同也一一对应）
    truth = _attach_occ(truth)
    pred  = _attach_occ(pred)

    join_keys = use_keys + ["__occ"]

    merged = truth.merge(
        pred[join_keys + [c for c in ["订单类型","prob_开线"] if c in pred.columns]],
        on=join_keys, how="inner", suffixes=("_truth","_pred")
    )

    # 计算命中（1/0）
    if "订单类型_truth" in merged.columns and "订单类型_pred" in merged.columns:
        merged["命中"] = (merged["订单类型_truth"] == merged["订单类型_pred"]).astype(int)
    else:
        merged["命中"] = np.nan

    # 评估
    unmatched_truth = len(truth) - len(merged)
    unmatched_pred  = len(pred)  - len(merged)

    if "订单类型_truth" in merged.columns and "订单类型_pred" in merged.columns:
        y_true = merged["订单类型_truth"].map({"接力":0, "开线":1})
        y_pred = merged["订单类型_pred"].map({"接力":0, "开线":1})
        mask = y_true.isin([0,1]) & y_pred.isin([0,1])
        y_true = y_true[mask].astype(int)
        y_pred = y_pred[mask].astype(int)

        if len(y_true):
            acc = (y_true == y_pred).mean()
            cm  = confusion_matrix(y_true, y_pred, labels=[0,1])
            rep = classification_report(
                y_true, y_pred,
                labels=[0,1],
                target_names=["接力(0)","开线(1)"],
                digits=4
            )
        else:
            acc = float("nan")
            cm  = np.array([[0,0],[0,0]])
            rep = "（无可比对样本）"
    else:
        acc = float("nan")
        cm  = np.array([[0,0],[0,0]])
        rep = "（无法计算，缺少预测或真值列）"

    # 导出 compare（优先 prob 排序，命中列包含在内）
    order_front = [c for c in KEEP_COLS if c in merged.columns]
    order_tail  = [c for c in merged.columns if c not in order_front]
    out_df = merged[order_front + order_tail].copy()

    sort_cols = ["prob_开线"] if "prob_开线" in out_df.columns else []
    if sort_cols:
        out_df.sort_values(sort_cols, ascending=False, inplace=True, ignore_index=True)

    out_df.to_excel(compare_out, index=False)

    # 日志
    log("\n" + "="*60)
    log(f"=== {title} ===")
    log(f"- truth条数：{len(truth)}   预测条数：{len(pred)}   对齐后：{len(merged)}")
    log(f"- truth 未对齐：{unmatched_truth}  |  predict 未对齐：{unmatched_pred}")
    log(f"- accuracy：{acc:.4f}" if np.isfinite(acc) else "- accuracy：NA")
    log(f"- 混淆矩阵（行=真实, 列=预测）：{cm.tolist()}")
    log("- 分类报告：")
    log(rep)
    log(f"- 详细对比表：{compare_out}")

def main():
    if SUMMARY_TXT.exists():
        SUMMARY_TXT.unlink()
    log("===== Truth 生成与预测对比=====")

    # 1) 生成 truth_day / truth_night（只保留有标签 + 附 __occ）
    read_truth_and_split()

    # 2) 分别与预测逐条对齐并生成 compare（内含“命中”=1/0）
    _one_to_one_compare(TRUTH_DAY_OUT,   PRED_DAY_FILE,   COMPARE_DAY_OUT,   "白班 对比")
    _one_to_one_compare(TRUTH_NIGHT_OUT, PRED_NIGHT_FILE, COMPARE_NIGHT_OUT, "夜班 对比")

if __name__ == "__main__":
    main()
    gc.collect()
    sys.exit(0)