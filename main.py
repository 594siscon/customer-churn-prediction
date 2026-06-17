import os
import pandas as pd
from src.data_preprocessor import xlsx_to_csv, prepare_modeling_data
from src.rfm_analyzer import calculate_rfm_scores, run_kmeans_clustering
from src.model_trainer import train_churn_model
from src.visualizer import (
    plot_rfm_distributions,
    plot_kmeans_elbow,
    plot_kmeans_profile,
    plot_kmeans_3d_static,
    plot_kmeans_3d_interactive,
    plot_pairwise_scatters,
    plot_feature_importances,
    plot_decision_tree,
    plot_confusion_matrices,
    plot_roc_curves
)

def main():
    print("=" * 60)
    print("   電商客戶價值分析 (RFM) 與客戶流失預測 (Random Forest) Pipeline")
    print("=" * 60)

    # 1. 定義路徑
    base_dir = r'c:\Users\user\Desktop\RFM'
    data_dir = os.path.join(base_dir, 'data')
    outputs_dir = os.path.join(base_dir, 'outputs')

    # 數據檔案路徑
    xlsx_path = os.path.join(data_dir, 'E Commerce Dataset.xlsx')
    raw_csv_path = os.path.join(data_dir, 'E Commerce Dataset.csv')
    processed_csv_path = os.path.join(data_dir, 'E Commerce Dataset Processed.csv')
    train_path = os.path.join(data_dir, 'train_set.csv')
    test_path = os.path.join(data_dir, 'test_set.csv')

    # 圖表輸出路徑
    dist_plot_path = os.path.join(outputs_dir, 'rfm_distributions.png')
    elbow_plot_path = os.path.join(outputs_dir, 'kmeans_elbow.png')
    profile_plot_path = os.path.join(outputs_dir, 'kmeans_profile_bars.png')
    static_3d_plot_path = os.path.join(outputs_dir, 'kmeans_3d_scatter.png')
    interactive_3d_html_path = os.path.join(outputs_dir, 'kmeans_3d_interactive.html')
    feat_imp_plot_path = os.path.join(outputs_dir, 'feature_importance.png')
    tree_plot_path = os.path.join(outputs_dir, 'decision_tree_structure.png')
    cm_plot_path = os.path.join(outputs_dir, 'confusion_matrix_comparison.png')
    roc_plot_path = os.path.join(outputs_dir, 'roc_curves_comparison.png')

    # 2. 數據轉換 (xlsx -> csv)
    if not os.path.exists(raw_csv_path):
        xlsx_to_csv(xlsx_path, raw_csv_path)
    else:
        print(f"[步驟 1] 發現已轉換的 CSV，載入：'{raw_csv_path}'")

    df_raw = pd.read_csv(raw_csv_path)

    # 3. 計算 RFM 分數
    print("\n[步驟 2] 計算 RFM 得分與客群分類...")
    df_rfm = calculate_rfm_scores(df_raw)
    
    # 繪製 RFM 欄位分佈圖
    plot_rfm_distributions(df_rfm, dist_plot_path)

    # 4. 進行 K-Means 聚類
    print("\n[步驟 3] 執行 K-Means 聚類分析 (K=5)...")
    df_clustered, k_range, wcss = run_kmeans_clustering(df_rfm, n_clusters=5)
    
    # 儲存更新後的原始 CSV（包含 RFM 得分與聚類標籤）
    df_clustered.to_csv(raw_csv_path, index=False, encoding='utf-8-sig')
    print(f"聚類分析完成，更新後的資料已儲存回：'{raw_csv_path}'")

    # 繪製聚類相關圖表
    print("正在產生聚類分析圖表...")
    plot_kmeans_elbow(k_range, wcss, elbow_plot_path)
    plot_kmeans_profile(df_clustered, profile_plot_path)
    plot_kmeans_3d_static(df_clustered, static_3d_plot_path)
    plot_kmeans_3d_interactive(df_clustered, interactive_3d_html_path)
    plot_pairwise_scatters(df_clustered, outputs_dir)

    # 5. 數據集分割與特徵工程
    print("\n[步驟 4] 保留特徵工程，分割訓練集與測試集 (80/20)...")
    train_df, test_df = prepare_modeling_data(raw_csv_path, processed_csv_path, train_path, test_path)

    # 6. 缺失值防漏填補與模型訓練 (GridSearchCV 5折交叉驗證)
    print("\n[步驟 5] 執行防漏缺失值填補與隨機森林 GridSearchCV 調參...")
    rf_model, feature_names, feat_imp_df, stats = train_churn_model(train_path, test_path)

    # 輸出訓練統計資訊
    print(f"訓練集維度: {stats['train_shape']} | 測試集維度: {stats['test_shape']}")
    print(f"隨機森林最佳超參數組合 (F1-score 最佳化): {stats['best_params']}")
    print("缺失值填補策略（僅擬合訓練集統計量）：")
    for col, strategy in stats['imputed_stats'].items():
        print(f"  * {col}: {strategy}")
    print(f"驗證缺失值是否清除：訓練集剩餘 = {stats['missing_train']}，測試集剩餘 = {stats['missing_test']}")

    # 輸出評估指標
    print(f"\n[步驟 6] 兩模型效能指標對比：")
    print(f"  * 決策樹測試集準確度 (DT Accuracy): {stats['dt_accuracy']:.4%}")
    print(f"  * 隨機森林測試集準確度 (RF Accuracy): {stats['rf_accuracy']:.4%}")
    print(f"  * 決策樹測試集 AUC (DT AUC): {stats['dt_auc']:.4f}")
    print(f"  * 隨機森林測試集 AUC (RF AUC): {stats['rf_auc']:.4f}")
    
    print("\n=== 決策樹模型分類報告 ===")
    print(stats['dt_report'])
    print("決策樹混淆矩陣:")
    print(stats['dt_cm'])
    
    print("\n=== 隨機森林模型分類報告 ===")
    print(stats['rf_report'])
    print("隨機森林混淆矩陣:")
    print(stats['rf_cm'])

    print("\n隨機森林特徵重要性排序 (Feature Importance):")
    print(feat_imp_df[feat_imp_df['Importance'] > 0])

    # 7. 繪製模型相關圖表
    print("\n[步驟 7] 正在產生模型關聯圖表...")
    plot_feature_importances(feat_imp_df, feat_imp_plot_path)
    # 繪製混淆矩陣對比圖
    plot_confusion_matrices(stats['dt_cm'], stats['rf_cm'], cm_plot_path)
    # 繪製 ROC 曲線對比圖
    X_test = test_df[feature_names]
    y_test = test_df['Churn']
    plot_roc_curves(stats['dt_model'], rf_model, X_test, y_test, roc_plot_path)

    # 8. 預測並標記高價值流失客戶 (Predicted_Churn = 1 且 M_Score >= 3 或 F_Score >= 3)
    print("\n[步驟 8] 正在預測並標記測試集中的高價值流失客戶...")
    test_df_marked = test_df.copy()
    y_pred_rf = rf_model.predict(X_test)
    
    # 新增預測流失欄位
    test_df_marked['Predicted_Churn'] = y_pred_rf
    
    # 高價值定義：消費金額(M)>=3 或 消費頻率(F)>=3，且被模型預測為流失的客戶
    high_value_cond = (test_df_marked['Predicted_Churn'] == 1) & (
        (test_df_marked['M_Score'] >= 3) | (test_df_marked['F_Score'] >= 3)
    )
    test_df_marked['High_Value_Churn_Alert'] = high_value_cond.astype(int)
    
    # 儲存回 test_set.csv (測試集此時已包含特徵、RFM欄位、預測結果與高價值預警標籤)
    test_df_marked.to_csv(test_path, index=False, encoding='utf-8-sig')
    print(f"高價值流失客戶已成功標記，並更新儲存至：'{test_path}'")
    
    # 控制台統計資訊
    total_predicted = int(y_pred_rf.sum())
    total_alert = int(test_df_marked['High_Value_Churn_Alert'].sum())
    print(f"  * 測試集被模型預測為流失客戶數：{total_predicted} 人")
    print(f"  * 其中 M>=3 或 F>=3 的高價值流失預警客戶數：{total_alert} 人")

    print("\n" + "=" * 60)
    print("  Pipeline 執行完成！所有輸出皆已分類整理：")
    print(f"  * 數據檔案目錄：'{data_dir}'")
    print(f"  * 可視化圖表目錄：'{outputs_dir}'")
    print("=" * 60)

if __name__ == '__main__':
    main()
