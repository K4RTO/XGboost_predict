import os, sys, gc, json
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"

from pathlib import Path
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import precision_recall_curve, classification_report, confusion_matrix

warnings.filterwarnings("ignore", message=r"Glyph .* missing from font")
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

BASE_DIR   = Path(__file__).resolve().parents[1]
DATA_DIR   = BASE_DIR / "Data"
OUTPUT_DIR = BASE_DIR / "Output"
MODEL_DIR  = BASE_DIR / "model"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DAY_DATA   = DATA_DIR / "day_data.xlsx"
NIGHT_DATA = DATA_DIR / "night_data.xlsx"

# ===== 关键配置 =====
FEATURES     = ["Line", "Line代码", "Lot", "Model", "model2", "PN", "供应商", "品名", "品类"]
TARGET       = "订单类型"
SEGMENT_KEY  = "Line代码"        # 分段阈值的分组字段（可改为 "Line"）

TARGET_POS_LABEL = "开线"
TARGET_NEG_LABEL = "接力"
TARGET_PRECISION = 0.80          # 目标：开线精确率不低于 80%
MIN_SEG_SUPPORT  = 200           # 分段计算阈值的最小样本量（不足则用全局阈值）

# ============== XGBoost (GPU) ==============
try:
    import xgboost as xgb
    HAS_XGB = True
except Exception as e:
    HAS_XGB = False
    XGB_IMPORT_ERR = str(e)

def log_env():
    print("===== 运行环境（GPU ONLY）=====")
    if not HAS_XGB:
        print(f"[错误] 未能导入 xgboost：{XGB_IMPORT_ERR}")
        raise SystemExit(1)
    try:
        _ = xgb.train(
            {"tree_method": "hist", "device": "cuda", "objective": "binary:logistic", "base_score": 0.5},
            dtrain=xgb.DMatrix(np.array([[0.0],[1.0]], dtype=np.float32),
                               label=np.array([0,1], dtype=np.float32)),
            num_boost_round=1
        )
        print("XGBoost GPU 可用: True")
    except Exception as e:
        print("XGBoost GPU 可用: False")
        print(f"[错误] {e}")
        raise SystemExit(1)
    print("=============================\n")

# ============== 工具函数 ==============
def _safe_name(s: str) -> str:
    return "".join(ch for ch in str(s) if ch.isalnum() or ch in ("_", "-"))

def coalesce_duplicate_columns(df: pd.DataFrame, cols_to_fix):
    """将重复列名合并为单列：逐行取第一个非空值"""
    df = df.copy()
    for col in cols_to_fix:
        same = [c for c in df.columns if c == col]
        if len(same) > 1:
            tmp = df.loc[:, same]
            fused = tmp.bfill(axis=1).iloc[:, 0]
            df.drop(columns=same, inplace=True)
            df[col] = fused
    return df

def kfold_target_encode(train_df, cols, target_col, n_splits=5, smoothing=20, random_state=42):
    """K 折目标编码：返回训练集的 out-of-fold 编码 和 完整映射（供推理用）"""
    df = train_df.copy()
    df[cols] = df[cols].astype(str)
    y = df[target_col].values
    global_mean = float(df[target_col].mean())
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    te_map = {c: {} for c in cols}
    out = pd.DataFrame(index=df.index, columns=cols, dtype=np.float32)

    for tr_idx, val_idx in kf.split(df):
        tr = df.iloc[tr_idx]
        va = df.iloc[val_idx]
        for c in cols:
            stats = tr.groupby(c)[target_col].agg(['mean','count']).rename(columns={'mean':'m','count':'n'})
            stats['te'] = (stats['n']*stats['m'] + smoothing*global_mean) / (stats['n'] + smoothing)
            out.loc[va.index, c] = va[c].map(stats['te']).astype(np.float32)

    for c in cols:
        out[c] = out[c].fillna(global_mean).astype(np.float32)
        stats_full = df.groupby(c)[target_col].agg(['mean','count']).rename(columns={'mean':'m','count':'n'})
        stats_full['te'] = (stats_full['n']*stats_full['m'] + smoothing*global_mean) / (stats_full['n'] + smoothing)
        te_map[c] = {
            "global": global_mean,
            "map": {str(k): float(v) for k, v in stats_full['te'].to_dict().items()}
        }
    return out.astype(np.float32), te_map

def pick_threshold_by_precision(y_true, prob, target_precision=0.80):
    """在满足精确率下限的前提下，选召回最高的阈值；若达不到，退回0.5"""
    prec, rec, thr = precision_recall_curve(y_true, prob)
    cands = [(P, R, T) for P, R, T in zip(prec[:-1], rec[:-1], thr) if P >= target_precision]
    if cands:
        bestP, bestR, bestT = max(cands, key=lambda x: (x[1], x[0]))
        return float(bestT), float(bestP), float(bestR)
    else:
        return 0.5, float(prec[0]), float(rec[0])

def plot_and_save(imp_df, title, filename):
    plt.figure(figsize=(10, 6))
    df_plot = imp_df[["Feature", "Importance"]].copy()
    df_plot.plot(x="Feature", y="Importance", kind="bar", legend=False, ax=plt.gca())
    plt.ylabel("特征重要性（gain）")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()

# ============== 训练主逻辑 ==============
def train_and_evaluate_one_shift(df_raw: pd.DataFrame, shift_name: str):
    print("\n" + "=" * 60)
    print(f"=== {shift_name} 班次：GPU建模（XGBoost + 目标编码 + 分段阈值） ===")
    print("=" * 60)

    df = coalesce_duplicate_columns(df_raw, FEATURES + [SEGMENT_KEY]).copy()

    if TARGET not in df.columns:
        print(f"[错误] 缺少目标列 {TARGET}")
        return

    # 只保留有标签的数据
    df[TARGET] = df[TARGET].astype(str).str.strip()
    df = df[df[TARGET].isin([TARGET_POS_LABEL, TARGET_NEG_LABEL])].copy()
    if df.empty:
        print("[错误] 无有效标签数据。")
        return

    # 目标转为 0/1
    df["y"] = (df[TARGET] == TARGET_POS_LABEL).astype(int)

    # 保证特征存在
    for c in FEATURES:
        if c not in df.columns:
            df[c] = ""

    # 目标编码（在训练集上做 K 折 OOF，避免泄漏）
    te_X, te_map = kfold_target_encode(
        df[FEATURES + ["y"]].copy(),
        cols=FEATURES,
        target_col="y",
        n_splits=5,
        smoothing=20,
        random_state=42
    )

    # 训练/验证划分
    X_train, X_test, y_train, y_test = train_test_split(
        te_X.values.astype(np.float32),
        df["y"].values.astype(np.int32),
        test_size=0.3, random_state=42, stratify=df["y"].values
    )

    # 记录分段键（和 X_test 对齐）
    # seg_series = df[SEGMENT_KEY].astype(str)
    # seg_test = seg_series.iloc[y_train.shape[0]: y_train.shape[0] + y_test.shape[0]].reset_index(drop=True)

    idx = np.arange(len(df))
    idx_tr, idx_te = train_test_split(
        idx, test_size=0.3, random_state=42, stratify=df["y"].values
    )

    X_train = te_X.iloc[idx_tr].values.astype(np.float32)
    X_test = te_X.iloc[idx_te].values.astype(np.float32)
    y_train = df["y"].iloc[idx_tr].astype(int).values
    y_test = df["y"].iloc[idx_te].astype(int).values

    # 分段键与测试集严格对齐
    seg_test = df[SEGMENT_KEY].astype(str).iloc[idx_te].reset_index(drop=True)

    # XGBoost 训练
    pos = int(y_train.sum()); neg = int(len(y_train) - pos)
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    params = {
        "tree_method": "hist",
        "device": "cuda",
        "nthread": 4,
        "max_bin": 256,
        "max_depth": 8,
        "eta": 0.1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "base_score": 0.5,
        "scale_pos_weight": scale_pos_weight,
        "gamma": 4.0,
        "min_child_weight": 5.0,
        "max_delta_step": 2.0,
    }

    evals = [(dtrain, "train"), (dtest, "test")]
    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=1000,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=50
    )

    # 全局阈值（优先保证开线精确率）
    prob_test = bst.predict(dtest)
    bestT, bestP, bestR = pick_threshold_by_precision(y_test, prob_test, TARGET_PRECISION)

    # 分段阈值
    per_segment = {}
    probs_df = pd.DataFrame({
        "prob": prob_test,
        "y": y_test,
        "seg": seg_test.fillna("__UNK__").astype(str)
    })
    for seg, g in probs_df.groupby("seg"):
        if len(g) < MIN_SEG_SUPPORT:
            continue
        t, p, r = pick_threshold_by_precision(g["y"].values, g["prob"].values, TARGET_PRECISION)
        per_segment[str(seg)] = float(t)

    # 以全局阈值的预测，打印评估
    y_pred_tuned = (prob_test >= bestT).astype(int)
    cm_tuned = confusion_matrix(y_test, y_pred_tuned, labels=[0, 1])
    rep_tuned = classification_report(
        y_test, y_pred_tuned, labels=[0, 1], target_names=[f"{TARGET_NEG_LABEL}(0)", f"{TARGET_POS_LABEL}(1)"], digits=4
    )

    print("\n[XGBoost (GPU) 结果 @全局阈值]")
    print(cm_tuned)
    print(rep_tuned)

    # 保存模型 + 目标编码映射 + 阈值
    model_path = MODEL_DIR / f"{shift_name}_model.json"
    bst.save_model(str(model_path))

    te_path = MODEL_DIR / f"{shift_name}_te_map.json"
    with open(te_path, "w", encoding="utf-8") as f:
        json.dump(te_map, f, ensure_ascii=False, indent=2)

    thr_obj = {
        "threshold": float(bestT),
        "per_segment": per_segment,
        "segment_key": SEGMENT_KEY,
        "target_precision": float(TARGET_PRECISION)
    }
    thr_path = MODEL_DIR / f"{shift_name}_threshold.json"
    with open(thr_path, "w", encoding="utf-8") as f:
        json.dump(thr_obj, f, ensure_ascii=False, indent=2)

    print(f"[模型] 已保存：{model_path}")
    print(f"[目标编码映射] 已保存：{te_path}")
    print(f"[阈值配置] 已保存：{thr_path}")

    # 特征重要性（基于 gain）
    score_gain = bst.get_score(importance_type="gain")
    fmap = {f"f{i}": FEATURES[i] for i in range(len(FEATURES))}
    imp_items = [(fmap.get(k, k), v) for k, v in score_gain.items()]
    used = {k for k, _ in imp_items}
    imp_items += [(c, 0.0) for c in FEATURES if c not in used]
    imp_df = pd.DataFrame(imp_items, columns=["Feature", "Importance"]).sort_values("Importance", ascending=False)
    total = imp_df["Importance"].sum()
    imp_df["占比"] = 0.0 if total == 0 else imp_df["Importance"]/total

    csv_out  = OUTPUT_DIR / f"{_safe_name(shift_name)}_feature_importance.csv"
    xlsx_out = OUTPUT_DIR / f"{_safe_name(shift_name)}_feature_importance.xlsx"
    imp_df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_out, engine="openpyxl") as writer:
        imp_df.to_excel(writer, index=False, sheet_name="feature_importance")
    print(f"[导出] 特征重要性：{csv_out}")
    print(f"[导出] 特征重要性：{xlsx_out}")

    # 画图
    plt.figure(figsize=(10, 6))
    imp_df.plot(x="Feature", y="Importance", kind="bar", legend=False)
    plt.ylabel("特征重要性（gain）")
    plt.title(f"{shift_name} XGBoost(GPU) 特征重要性")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{_safe_name(shift_name)}_feature_importance.png", dpi=150)
    plt.close()

def main():
    log_env()

    if not DAY_DATA.exists() or not NIGHT_DATA.exists():
        print(f"[错误] 找不到数据文件：\n- {DAY_DATA}\n- {NIGHT_DATA}")
        return

    cat_dtype = {c: str for c in FEATURES + [TARGET, SEGMENT_KEY]}
    day_df   = pd.read_excel(DAY_DATA,   dtype=cat_dtype)
    night_df = pd.read_excel(NIGHT_DATA, dtype=cat_dtype)

    # 合并重复列，防止 groupby 报错
    day_df   = coalesce_duplicate_columns(day_df,   FEATURES + [SEGMENT_KEY])
    night_df = coalesce_duplicate_columns(night_df, FEATURES + [SEGMENT_KEY])

    print(f"白班数据量：{len(day_df)}")
    print(f"夜班数据量：{len(night_df)}")

    train_and_evaluate_one_shift(day_df, "白班")
    train_and_evaluate_one_shift(night_df, "夜班")

if __name__ == "__main__":
    main()
    plt.close("all")
    gc.collect()
    sys.exit(0)