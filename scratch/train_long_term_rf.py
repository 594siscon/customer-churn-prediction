import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 設定畫圖美學樣式與中文字型
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Segoe UI', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def main():
    processed_csv_path = r'c:\Users\user\Desktop\RFM\data\E Commerce Dataset Processed.csv'
    outputs_dir = r'c:\Users\user\Desktop\RFM\outputs'
    feat_imp_plot_path = os.path.join(outputs_dir, 'long_term_feature_importance.png')
    
    if not os.path.exists(processed_csv_path):
        print(f"找不到處理後的資料集: {processed_csv_path}")
        return
        
    df = pd.read_csv(processed_csv_path)
    
    # === 1. 標記長期客戶 (Tenure > 12.0) ===
    df['Is_Long_Term'] = (df['Tenure'] > 12.0).astype(int)
    
    print("【長期客戶標籤分佈】")
    print(df['Is_Long_Term'].value_counts())
    print(f"長期客戶佔比: {df['Is_Long_Term'].mean():.2%}\n")
    
    # === 2. 分割訓練集與測試集 (80/20) ===
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Is_Long_Term'])
    
    # === 3. 缺失值填補 (訓練集統計量) ===
    hour_mode = train_df['HourSpendOnApp'].mode()[0]
    train_df['HourSpendOnApp'] = train_df['HourSpendOnApp'].fillna(hour_mode)
    test_df['HourSpendOnApp'] = test_df['HourSpendOnApp'].fillna(hour_mode)
    
    train_df['CouponUsed'] = train_df['CouponUsed'].fillna(0)
    test_df['CouponUsed'] = test_df['CouponUsed'].fillna(0)
    
    for col in train_df.columns:
        if col not in ['HourSpendOnApp', 'CouponUsed', 'Is_Long_Term']:
            if pd.api.types.is_numeric_dtype(train_df[col]) and train_df[col].isnull().any():
                median_val = train_df[col].median()
                train_df[col] = train_df[col].fillna(median_val)
                test_df[col] = test_df[col].fillna(median_val)
                
    # === 4. 分離特徵與標籤 (防洩漏) ===
    non_feature_cols = [
        'CustomerID', 'R_Score', 'F_Score', 'M_Score', 'RFM_Group', 'RFM_Score', 'KMeans_Cluster', 
        'Churn', 'Tenure', 'Is_Long_Term', 'Predicted_Churn', 'High_Value_Churn_Alert'
    ]
    
    X_train = train_df.drop(columns=non_feature_cols, errors='ignore')
    y_train = train_df['Is_Long_Term']
    
    X_test = test_df[X_train.columns]
    y_test = test_df['Is_Long_Term']
    
    feature_names = list(X_train.columns)
    
    # === 5. 隨機森林二次調參 (拓展搜尋空間) ===
    print("正在執行隨機森林拓展超參數調優 (GridSearchCV, CV=5)...")
    rf_base = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # 將網格往更深、更多樹、限制更少方向擴展，探索最佳邊界之外
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [10, 12, 15],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [2, 4]
    }
    
    grid_search = GridSearchCV(
        estimator=rf_base,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_rf_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print("\n" + "=" * 60)
    print("           隨機森林最優調參參數 (二次調優)")
    print("=" * 60)
    print(f"最佳參數組合: {best_params}")
    
    # === 模型持久化存檔 ===
    import joblib
    models_dir = r'c:\Users\user\Desktop\RFM\models'
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, 'best_long_term_rf.joblib')
    features_save_path = os.path.join(models_dir, 'long_term_features.joblib')
    
    joblib.dump(best_rf_model, model_save_path)
    joblib.dump(feature_names, features_save_path)
    print(f"\n[長期模型存檔成功] 最優長期預測模型已保存至: '{model_save_path}'")
    print(f"[長期特徵清單存檔成功] 最優長期特徵名稱已保存至: '{features_save_path}'")
    
    # === 6. 模型評估 ===
    y_pred = best_rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "=" * 60)
    print("           二次調參隨機森林模型評估結果")
    print("=" * 60)
    print(f"測試集準確度 (Accuracy): {accuracy:.4%}")
    print("\n分類報告 (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=['非長期客戶 (0)', '長期客戶 (1)']))
    print("混淆矩陣 (Confusion Matrix):")
    print(cm)
    
    # === 7. 提取特徵重要性與繪圖 ===
    importances = best_rf_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\n隨機森林特徵重要性排序 (前15名):")
    print(feat_imp_df.head(15))
    
    print("\n正在重新生成特徵重要性圖表...")
    plt.figure(figsize=(10, 6), dpi=150)
    sns.barplot(
        x='Importance',
        y='Feature',
        data=feat_imp_df.head(15),
        palette='viridis',
        hue='Feature',
        legend=False
    )
    plt.title("預測長期潛力客戶之最優隨機森林特徵重要性 (Top 15)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("重要性分數", fontsize=11)
    plt.ylabel("特徵欄位", fontsize=11)
    plt.tight_layout()
    plt.savefig(feat_imp_plot_path, bbox_inches='tight')
    plt.close()
    print(f"特徵重要性條形圖已重新儲存至: {feat_imp_plot_path}")

if __name__ == '__main__':
    main()
