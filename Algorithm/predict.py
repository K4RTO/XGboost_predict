import os, sys, gc, json
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

BASE_DIR  = Path(__file__).resolve().parents[1]
DATA_DIR  = BASE_DIR / "Data"
MODEL_DIR = BASE_DIR / "model"
OUT_DIR   = BASE_DIR / "Result"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CALL_OFF_XLSX = DATA_DIR / "call_off.xlsx"
SHEET_NAME    = 0

# 训练使用的字段（与训练脚本一致）
FEATURES = ["Line", "Line代码", "Lot", "Model", "model2", "PN", "供应商", "品名", "品类"]

def log(msg: str):
    print(msg)
    with (OUT_DIR / "summary.txt").open("a", encoding="utf-8") as f:
        f.write(msg + "\n")

def detect_gpu():
    try:
        _ = xgb.train(
            {"tree_method": "hist", "device": "cuda", "objective": "binary:logistic", "base_score": 0.5},
            dtrain=xgb.DMatrix(np.array([[0.],[1.]], dtype=np.float32),
                               label=np.array([0,1], dtype=np.float32)),
            num_boost_round=1
        )
        return True, None
    except Exception as e:
        return False, str(e)

def parse_shift_to_daynight(series: pd.Series) -> pd.Series:
    s = series.copy()
    # 数字路径（Excel 序列）
    mask_num = pd.to_numeric(s, errors="coerce").notna()
    dt_num = pd.to_datetime(pd.to_numeric(s[mask_num], errors="coerce"),
                            unit="D", origin="1899-12-30", errors="coerce")
    # 字符串路径
    s_str = s[~mask_num].astype(str).str.strip().str.replace(r"[./]", "-", regex=True)
    dt_str = pd.to_datetime(s_str, errors="coerce")
    # 合并
    dt = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    dt.loc[mask_num] = dt_num
    dt.loc[~mask_num] = dt_str
    hour = dt.dt.hour
    return np.where(hour == 8, "白班", np.where(hour == 20, "夜班", "未知班次"))

def load_model(shift_name: str) -> xgb.Booster:
    path = MODEL_DIR / f"{shift_name}_model.json"
    if not path.exists():
        raise FileNotFoundError(f"未找到模型文件: {path}")
    bst = xgb.Booster()
    bst.load_model(str(path))
    return bst

def load_te_map(shift_name: str) -> dict:
    path = MODEL_DIR / f"{shift_name}_te_map.json"
    if not path.exists():
        raise FileNotFoundError(f"未找到目标编码映射: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))

def load_thresholds(shift_name: str) -> dict:
    path = MODEL_DIR / f"{shift_name}_threshold.json"
    if not path.exists():
        return {"threshold": 0.5, "per_segment": {}, "segment_key": "Line代码", "target_precision": 0.0}
    return json.loads(Path(path).read_text(encoding="utf-8"))

def apply_te_transform(df: pd.DataFrame, features, te_map: dict) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    for c in features:
        vals = df.get(c, "").astype(str).fillna("")
        m = te_map.get(c, {})
        g = m.get("global", 0.5)
        d = m.get("map", {})
        X[c] = vals.map(lambda v: d.get(str(v), g)).astype(np.float32)
    return X.astype(np.float32)

def predict_for_shift(df_shift: pd.DataFrame, shift_name: str, out_file: Path):
    if df_shift.empty:
        log(f"[{shift_name}] 无数据，跳过。")
        return

    # 保留原始行序（不排序）
    bst    = load_model(shift_name)
    te_map = load_te_map(shift_name)
    thrObj = load_thresholds(shift_name)

    use_cols = [c for c in FEATURES if c in df_shift.columns]
    te_X = apply_te_transform(df_shift, use_cols, te_map)
    prob = bst.predict(xgb.DMatrix(te_X.values))

    # 分段阈值
    seg_key = thrObj.get("segment_key", "Line代码")
    per_seg = thrObj.get("per_segment", {})
    globalT = float(thrObj.get("threshold", 0.5))
    seg_vals = df_shift.get(seg_key, "__UNK__").astype(str).fillna("__UNK__")
    thr_vec = np.array([per_seg.get(str(s), globalT) for s in seg_vals], dtype=np.float32)
    pred = (prob >= thr_vec).astype(int)

    # 不打乱行顺序，不强制 ROW_ID
    out = df_shift.copy()
    if "订单类型" in out.columns:
        out.rename(columns={"订单类型": "原始订单类型"}, inplace=True)
    out["prob_开线"] = prob
    out["订单类型"] = np.where(pred == 1, "开线", "接力")
    out["班次"] = shift_name

    out.to_excel(out_file, index=False)
    log(f"[输出] {shift_name} -> {out_file} （{len(out)} 行）")

def main():
    ok, err = detect_gpu()
    log("===== 预测环境检查 =====")
    if not ok:
        log(f"XGBoost GPU 可用: False -> {err}")
        return
    if not CALL_OFF_XLSX.exists():
        log(f"[错误] 未找到 {CALL_OFF_XLSX}")
        return

    raw = pd.read_excel(CALL_OFF_XLSX, sheet_name=SHEET_NAME)

    # 解析“班次”→ 白/夜
    shift_txt = parse_shift_to_daynight(raw["班次"]) if "班次" in raw.columns else "未知班次"
    raw["_白夜班"] = shift_txt

    day_df   = raw[raw["_白夜班"] == "白班"].copy()
    night_df = raw[raw["_白夜班"] == "夜班"].copy()

    predict_for_shift(day_df,   "白班", OUT_DIR / "predict_day.xlsx")
    predict_for_shift(night_df, "夜班", OUT_DIR / "predict_night.xlsx")

if __name__ == "__main__":
    main()
    gc.collect()
    sys.exit(0)