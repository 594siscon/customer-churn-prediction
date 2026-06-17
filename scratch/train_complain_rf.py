import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import shap

# 設定美學與中文字型
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Segoe UI', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def main():
    # 1. 定義路徑
    processed_csv_path = r'c:\Users\user\Desktop\RFM\data\E Commerce Dataset Processed.csv'
    outputs_dir = r'c:\Users\user\Desktop\RFM\outputs'
    
    feat_imp_path = os.path.join(outputs_dir, 'complain_feature_importance.png')
    cm_plot_path = os.path.join(outputs_dir, 'complain_confusion_matrix.png')
    shap_summary_path = os.path.join(outputs_dir, 'complain_shap_summary.png')
    shap_dep_path = os.path.join(outputs_dir, 'complain_shap_dependence.png')
    
    if not os.path.exists(processed_csv_path):
        print(f"找不到處理後的資料集: {processed_csv_path}")
        return
        
    df = pd.read_csv(processed_csv_path)
    
    # 檢查目標變數 Complain 的分佈
    complain_counts = df['Complain'].value_counts()
    complain_pct = df['Complain'].value_counts(normalize=True)
    print("=" * 60)
    print("   投訴預測隨機森林模型建立與分析")
    print("=" * 60)
    print(f"目標變數 Complain 分佈情況:")
    print(f"  * 未投訴 (0): {complain_counts[0]} 筆 ({complain_pct[0]:.2%})")
    print(f"  * 有投訴 (1): {complain_counts[1]} 筆 ({complain_pct[1]:.2%})")
    
    # 2. 切分訓練與測試集 (80/20) 并保持 Stratify
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['Complain']
    )
    
    # 3. 缺失值填補 (以防洩漏：使用訓練集統計量)
    hour_mode = train_df['HourSpendOnApp'].mode()[0]
    train_df['HourSpendOnApp'] = train_df['HourSpendOnApp'].fillna(hour_mode)
    test_df['HourSpendOnApp'] = test_df['HourSpendOnApp'].fillna(hour_mode)
    
    train_df['CouponUsed'] = train_df['CouponUsed'].fillna(0)
    test_df['CouponUsed'] = test_df['CouponUsed'].fillna(0)
    
    for col in train_df.columns:
        if col not in ['HourSpendOnApp', 'CouponUsed', 'Complain']:
            if pd.api.types.is_numeric_dtype(train_df[col]) and train_df[col].isnull().any():
                median_val = train_df[col].median()
                train_df[col] = train_df[col].fillna(median_val)
                test_df[col] = test_df[col].fillna(median_val)
                
    # 4. 分離 X 與 y (防洩漏排除 CustomerID, Churn 等)
    # Churn 是流失狀態，流失通常發生在投訴之後，不能拿來當預測特徵
    non_feature_cols = [
        'CustomerID', 'R_Score', 'F_Score', 'M_Score', 'RFM_Group', 'RFM_Score', 'KMeans_Cluster', 
        'Churn', 'Complain', 'Predicted_Churn', 'High_Value_Churn_Alert'
    ]
    
    X_train = train_df.drop(columns=non_feature_cols, errors='ignore')
    y_train = train_df['Complain']
    X_test = test_df[X_train.columns]
    y_test = test_df['Complain']
    
    # 5. 訓練基準決策樹 (DT) 以作對照
    print("\n正在訓練基準決策樹模型...")
    dt_model = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    y_prob_dt = dt_model.predict_proba(X_test)[:, 1]
    
    # 6. 使用 GridSearchCV 進行隨機森林 (RF) 調參
    print("\n正在進行隨機森林 (Random Forest) 網格搜尋調參...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [8, 12, 15],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4]
    }
    
    rf_base = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    rf_best = grid_search.best_estimator_
    y_pred_rf = rf_best.predict(X_test)
    y_prob_rf = rf_best.predict_proba(X_test)[:, 1]
    
    print(f"隨機森林最佳超參數: {grid_search.best_params_}")
    
    # === 模型持久化存檔 ===
    import joblib
    models_dir = r'c:\Users\user\Desktop\RFM\models'
    os.makedirs(models_dir, exist_ok=True)
    
    model_save_path = os.path.join(models_dir, 'best_complain_rf.joblib')
    features_save_path = os.path.join(models_dir, 'complain_features.joblib')
    
    joblib.dump(rf_best, model_save_path)
    joblib.dump(list(X_train.columns), features_save_path)
    
    print(f"\n[投訴模型存檔成功] 最優投訴預測模型已保存至: '{model_save_path}'")
    print(f"[投訴特徵清單存檔成功] 最優投訴特徵名稱已保存至: '{features_save_path}'")
    
    # 7. 評估兩模型效能
    dt_acc = accuracy_score(y_test, y_pred_dt)
    rf_acc = accuracy_score(y_test, y_pred_rf)
    dt_auc = roc_auc_score(y_test, y_prob_dt)
    rf_auc = roc_auc_score(y_test, y_prob_rf)
    
    print("\n" + "=" * 50)
    print("   模型效能評估對比")
    print("=" * 50)
    print(f"1. 測試集準確度 (Accuracy):")
    print(f"   * 基準決策樹 (DT): {dt_acc:.2%}")
    print(f"   * 最優隨機森林 (RF): {rf_acc:.2%}")
    print(f"2. 測試集 AUC-ROC 得分:")
    print(f"   * 基準決策樹 (DT): {dt_auc:.4f}")
    print(f"   * 最優隨機森林 (RF): {rf_auc:.4f}")
    
    print("\n基準決策樹 (DT) 分類報告:")
    print(classification_report(y_test, y_pred_dt))
    print("\n最優隨機森林 (RF) 分類報告:")
    print(classification_report(y_test, y_pred_rf))
    
    # 8. 繪製混淆矩陣
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    
    plt.figure(figsize=(6, 5), dpi=150)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=['無投訴', '有投訴'], yticklabels=['無投訴', '有投訴'])
    plt.title("最優隨機森林 (RF) 混淆矩陣", fontsize=12, fontweight='bold')
    plt.xlabel("預測類別")
    plt.ylabel("實際類別")
    
    plt.tight_layout()
    plt.savefig(cm_plot_path, bbox_inches='tight')
    plt.close()
    print(f"最優隨機森林混淆矩陣圖已儲存至: {cm_plot_path}")
    
    # 9. 繪製隨機森林特徵重要性 (Feature Importance)
    importances = rf_best.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 15
    
    plt.figure(figsize=(10, 6), dpi=150)
    sns.barplot(
        x=importances[indices[:top_n]],
        y=np.array(X_train.columns)[indices[:top_n]],
        palette='viridis',
        hue=np.array(X_train.columns)[indices[:top_n]],
        legend=False
    )
    plt.title("最優隨機森林投訴預測 - 前 15 大特徵重要性排序", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("特徵重要性得分")
    plt.ylabel("特徵欄位")
    plt.tight_layout()
    plt.savefig(feat_imp_path, bbox_inches='tight')
    plt.close()
    print(f"特徵重要性條形圖已儲存至: {feat_imp_path}")
    
    # 10. 計算 TreeSHAP 並繪製全局與相依性圖
    print("\n正在計算 TreeSHAP 值以進行特徵歸因分析...")
    explainer = shap.TreeExplainer(rf_best)
    shap_values = explainer(X_test)
    
    # 若是三維的 shap_values (二元分類)，提取 class 1 (投訴) 的 SHAP 值
    if len(shap_values.shape) == 3:
        shap_vals_class1 = shap_values[:, :, 1]
    else:
        shap_vals_class1 = shap_values
        
    # 繪製 SHAP 全局摘要圖 (Beeswarm)
    plt.figure(figsize=(10, 6), dpi=150)
    shap.plots.beeswarm(shap_vals_class1, max_display=12, show=False)
    plt.title("最優隨機森林投訴預測 SHAP 全局摘要圖", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(shap_summary_path, bbox_inches='tight')
    plt.close()
    print(f"SHAP 全局摘要圖已儲存至: {shap_summary_path}")
    
    # 繪製指定特徵的相依性圖 (1x3)
    target_feats = ['Tenure', 'WarehouseToHome', 'Gender_Male']
    print(f"正在針對 {target_feats} 繪製 SHAP 相依性圖...")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 5), dpi=150)
    fig.suptitle("最優隨機森林投訴預測 - 關鍵特徵 SHAP 相依性分析與臨界點探索", fontsize=14, fontweight='bold', y=1.02)
    
    # 英文對照中文標題
    chinese_titles = {
        'Tenure': '客戶年資 (Tenure)',
        'WarehouseToHome': '配送距離 (WarehouseToHome)',
        'Gender_Male': '性別：男性 (Gender_Male)'
    }
    
    for idx, feat in enumerate(target_feats):
        ax = axes[idx]
        if feat == 'WarehouseToHome':
            # 排除 WarehouseToHome > 120 的極端異常點以避免 X 軸被拉得過寬，將 mask 轉成 numpy.ndarray 以相容 slicer
            mask = (X_test['WarehouseToHome'] <= 120).values
            shap.plots.scatter(shap_vals_class1[mask, feat], color=shap_vals_class1[mask, feat], ax=ax, show=False)
        else:
            shap.plots.scatter(shap_vals_class1[:, feat], color=shap_vals_class1[:, feat], ax=ax, show=False)
        title_text = chinese_titles.get(feat, feat)
        ax.set_title(f"{idx+1}. {title_text}", fontsize=11, fontweight='bold')
        ax.set_ylabel("SHAP 值 (對投訴機率的邊際貢獻)", fontsize=9)
        ax.axhline(0, color='red', linestyle='--', alpha=0.6)
        
    plt.tight_layout()
    plt.savefig(shap_dep_path, bbox_inches='tight')
    plt.close()
    print(f"SHAP 相依性圖已儲存至: {shap_dep_path}")

if __name__ == '__main__':
    main()
