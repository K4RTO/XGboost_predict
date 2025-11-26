# 产线订单类型预测与验证

端到端流程，用于对产线订单类型（“开线”/“接力”）进行数据预处理、白/夜班独立建模（XGBoost GPU）、预测推理与结果验证。

## 目录结构
```
项目根目录/
├── Algorithm/           # 模型训练与推理
│   ├── classification.py   # 白/夜班目标编码 + XGBoost(GPU) 训练
│   └── predict.py          # 按班次载入模型并生成预测
├── Process/            # 数据处理与验证
│   ├── data_classifier.py  # 可选：按日期窗口切出 call_off / truth / data
│   ├── data_process.py     # 清洗训练数据，生成 day_data/night_data
│   └── data_verify.py      # 预测结果与真值逐条对齐并评估
├── Data/               # 训练/预测/真值数据
│   ├── origin_data.xlsx
│   ├── data.xlsx / Processed_data.xlsx / day_data.xlsx / night_data.xlsx
│   ├── call_off.xlsx        # 待预测
│   └── truth.xlsx           # 真值（含标签）
├── model/              # 白/夜班模型、目标编码映射、阈值
├── Output/             # 训练阶段特征重要性、开线率对比图表与日志
├── Result/             # 预测输出、真值拆分、对比表、验证日志
├── requirements.txt
└── README.md
```

## 环境准备
- Python 3.9+（推荐使用本地虚拟环境）
- GPU/CUDA 可用（训练与预测脚本会自动检测）
```bash
pip install -r requirements.txt
```

## 运行流程
1) （可选）切分当期数据  
```bash
python Process/data_classifier.py
```  
按配置的日期窗口，将原始数据拆为 `call_off.xlsx`（预测用）、`truth.xlsx`（验证用）、`data.xlsx`（历史训练）。

2) 数据预处理  
```bash
python Process/data_process.py
```  
清洗训练数据，解析班次为白/夜，导出 `Processed_data.xlsx`、`day_data.xlsx`、`night_data.xlsx`。

3) 模型训练（GPU）  
```bash
python Algorithm/classification.py
```  
对白班、夜班分别训练，保存模型、目标编码映射、全局/分段阈值，并输出特征重要性图表/表格。

4) 预测  
```bash
python Algorithm/predict.py
```  
读取 `call_off.xlsx`，按班次生成 `Result/predict_day.xlsx`、`Result/predict_night.xlsx`，日志写入 `Result/summary.txt`。

5) 验证  
```bash
python Process/data_verify.py
```  
对齐真值与预测，计算准确率、混淆矩阵和分类报告，输出 `compare_day.xlsx`、`compare_night.xlsx`，日志追加到 `Result/summary.txt`。

## 输出说明
- `Result/`：预测结果（predict_day/night）、真值拆分（truth_day/night）、对比表（compare_day/night）、验证日志（summary.txt）。  
- `Output/`：训练阶段产物（特征重要性 CSV/XLSX/PNG、开线率对比图表、规律挖掘结果、训练日志）。

## 备注
- `summary.txt` 采用追加写入，如需重新跑一轮可先清空该文件。  
- 文档已去除任何真实公司名称，可直接在内部或外部环境使用。***
