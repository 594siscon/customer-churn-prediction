# 客戶流失/長期/投訴預測專案

本專案旨在提供一套端到端的電商數據分析與多向機器學習預測 Pipeline（客戶流失預警、潛力長期客戶預測、以及客戶投訴行為預測）。專案已完成模組化重構，採用標準的軟體開發目錄結構，並在所有數據填補與訓練流程中嚴防**資訊洩漏 (Data Leakage)**。

---

## 專案目錄結構

重構並整理後的目錄結構如下所示：

```text
RFM/
├── data/                               # 數據儲存目錄
│   ├── E Commerce Dataset.xlsx          # 原始電商 Excel 資料源
│   ├── E Commerce Dataset.csv           # 轉換後的原始 CSV 數據
│   ├── E Commerce Dataset Processed.csv # 特徵工程與編碼後的建模數據
│   ├── train_set.csv                    # 分割出的訓練集 (保留原始缺失值防洩漏)
│   └── test_set.csv                     # 分割出的測試集 (含隨機森林預測與高價值流失標籤)
├── src/                                # 模組化核心業務邏輯
│   ├── __init__.py
│   ├── data_preprocessor.py             # 模組 1：資料格式轉換、特徵獨熱編碼、80/20 分割
│   ├── rfm_analyzer.py                  # 模組 2：RFM 得分計算、KMeans 聚類分析
│   ├── churn_predictor.py               # 模組 3：隨機森林/決策樹建模、缺失值防漏填補
│   └── visualizer.py                    # 模組 4：統一管理所有圖表與 3D 互動可視化邏輯
├── outputs/                            # 圖像與分析報告輸出目錄
│   ├── rfm_distributions.png            # R, F, M 數據分布圖
│   ├── kmeans_elbow.png                # KMeans 最佳群數手肘法圖 (最佳群數 = 5)
│   ├── kmeans_profile_bars.png         # KMeans 各客群特徵 Z-Score 偏差柱狀圖
│   ├── kmeans_3d_scatter.png           # KMeans 靜態 3D 客群散點圖
│   ├── kmeans_3d_interactive.html       # KMeans 互動式 3D 散點網頁檔 (Plotly HTML)
│   ├── rfm_scatter_rf.png               # Recency - Frequency 2D 關係散點圖
│   ├── rfm_scatter_fm.png               # Frequency - Monetary 2D 關係散點圖
│   ├── rfm_scatter_rm.png               # Recency - Monetary 2D 關係散點圖
│   ├── feature_importance.png           # 客戶流失預測隨機森林特徵重要性圖
│   ├── confusion_matrix_comparison.png  # 兩預測模型流失預測之混淆矩陣對比圖
│   ├── roc_curves_comparison.png        # 兩模型流失預測之 ROC 曲線與 AUC 對比圖
│   ├── shap_summary.png                 # 客戶流失預測之 SHAP 全局摘要圖 (Beeswarm)
│   ├── shap_waterfall_example.png       # 代表性流失客戶的 SHAP 局部歸因瀑布圖
│   ├── top5_features_distributions.png  # 流失模型 Top 5 特徵在兩數據集的分佈對照圖
│   ├── long_term_decision_tree.png      # 潛力長期客戶預測之決策樹結構圖
│   ├── long_term_feature_importance.png # 潛力長期客戶隨機森林特徵重要性圖
│   ├── long_term_shap_summary.png       # 潛力長期客戶 SHAP 全局摘要圖 (Beeswarm)
│   ├── long_term_shap_dependence.png    # 長期客戶三大特徵臨界值 (165元 / 3地址 / 15公里) 探索圖
│   ├── preferedordercat_shap_summary.png # 商品偏好類別 (PreferedOrderCat_) SHAP 全局摘要圖
│   ├── preferedordercat_shap_dependence/ # 商品偏好各虛擬欄位的個別 SHAP 相依性散點圖目錄
│   ├── complain_confusion_matrix.png    # 最優隨機森林投訴預測混淆矩陣圖
│   ├── complain_feature_importance.png  # 投訴預測之隨機森林特徵重要性排序圖
│   ├── complain_shap_summary.png        # 投訴預測之 SHAP 全局摘要圖 (Beeswarm)
│   └── complain_shap_dependence.png     # 投訴預測之關鍵驅動特徵 (Cashback / Tenure) 相依性探索圖
├── scratch/                            # 深度歸因分析與長期/投訴預測輔助腳本
│   ├── analyze_top5.py                  # 驗證流失模型前五大特徵在訓練集與測試集的分佈一致性
│   ├── calculate_shap.py                # 客戶流失預測模型 TreeSHAP 計算與歸因分析
│   ├── train_long_term_dt.py            # 潛力長期客戶決策樹調參與結構可視化
│   ├── train_long_term_rf.py            # 潛力長期客戶隨機森林二次調優與特徵權重分析
│   ├── calculate_long_term_shap.py      # 長期客戶 TreeSHAP 值與三大特徵臨界點分析
│   ├── calculate_preferedordercat_shap.py # 商品偏好類別一熱編碼變數的 SHAP 依賴性繪製
│   └── train_complain_rf.py             # 投訴預測 (Complain) 隨機森林調參、評估與 SHAP 臨界點分析
├── main.py                             # 端到端一鍵運行 Pipeline 入口
└── README.md                           # 本說明文件
```

---

## 核心 Pipeline 與建模功能介紹

1. **`data_preprocessor.py`**
   * 將原始 Excel 轉為 CSV，清理 CustomerID、RFM 分群等特徵，防範資訊洩漏。
   * 將類別型變數進行 One-Hot 編碼，並將數據按 80/20 比例分割為 `train_set.csv` 與 `test_set.csv`。
2. **`rfm_analyzer.py`**
   * 計算近老度、頻率、金額得分 (R_Score, F_Score, M_Score) 並進行客群分群。
   * 執行 KMeans 分群分析 (K=5)，更新並保存聚類標籤的原始 CSV。
3. **`churn_predictor.py`**
   * 載入分割數據。**僅在訓練集上擬合缺失值填補值**（眾數、中位數、常數0），並套用到兩集上，徹底防止資訊洩漏。
   * 採用隨機森林 (Random Forest)，使用 `GridSearchCV` 進行 5 折交叉驗證超參數優化（流失預測測試集 Accuracy 達 **92.45%**，AUC 達 **0.9692**）。
4. **`visualizer.py`**
   * 負責所有的資料分布圖、2D散點、3D互動式 HTML、特徵重要性、ROC 曲線與混淆矩陣對比圖的生成。

---

## 機器學習雙向與多維度預測成果

### 1. 高價值流失預警標記
在 `main.py` 運行後，會在 [test_set.csv](file:///c:/Users/user/Desktop/RFM/data/test_set.csv) 中新增：
*   `Predicted_Churn`：最優隨機森林預測是否流失 (1/0)。
*   `High_Value_Churn_Alert`：高價值流失預警標籤 (1/0)。當預測流失且消費金額 (M_Score) $\ge$ 3 或消費頻率 (F_Score) $\ge$ 3 時標記為 1。這為營運團隊提供了精準的**挽回黃金名單**。

### 2. 潛力長期客戶預測與篩選規則
排除會造成洩漏的特徵後，調參預測年資大於 1 年 (`Tenure > 12.0` 個月) 的潛力長期客戶。經 TreeSHAP 分析，業務篩選臨界值為：
*   **回饋金臨界值 (`CashbackAmount`)**：平均回饋金需達 **165 元以上**，對長期忠誠度產生強烈的正向拉力。
*   **登記地址數臨界值 (`NumberOfAddress`)**：登記地址數需達 **3 個或以上**，代表較高的平台生活融入度與轉換成本。
*   **配送距離臨界值 (`WarehouseToHome`)**：配送距離控制在 **15 公里以內**。

### 3. 客戶投訴行為預測與歸因 (Complain)
以 `Complain` (1/0) 作為目標變數，採用隨機森林模型並搭配 `class_weight='balanced'` 處理類別不平衡，經網格搜尋二次調優後，**測試集 Accuracy 達到 89.17%，AUC 達 0.9315，且投訴預測精準率 (Precision) 達到了 95%**！
經 TreeSHAP 探索，驅動客訴的兩大核心要素為：
*   **回饋金金額 (`CashbackAmount`)**：回饋金高低直接與客訴呈負相關，高優惠能大幅沖淡投訴動機。
*   **客戶年資 (`Tenure`)**：新註冊客群相比老客，由於對平台的流程尚未熟悉或期待落差，更容易引發客訴。

---

## 運行指南

### 1. 執行核心 Pipeline (流失預測與分群)
在專案根目錄下，使用 Python 直接執行 `main.py` 即可一鍵運行完整的分析與建模流程：
```bash
python main.py
```

### 2. 執行深度 SHAP 與預測歸因分析
```bash
# 執行流失預測 SHAP 分析
python scratch/calculate_shap.py

# 執行長期客戶 SHAP 臨界值分析
python scratch/calculate_long_term_shap.py

# 執行投訴行為隨機森林預測與 SHAP 歸因分析
python scratch/train_complain_rf.py
```
所有的分析圖表將自動保存於 `outputs/` 目錄中。
