import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re

# 設定畫圖美學樣式與中文字型
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Segoe UI', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def main():
    processed_csv_path = r'c:\Users\user\Desktop\RFM\data\E Commerce Dataset Processed.csv'
    outputs_dir = r'c:\Users\user\Desktop\RFM\outputs'
    tree_plot_path = os.path.join(outputs_dir, 'long_term_decision_tree.png')
    
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
    
    # === 5. 決策樹超參數調優 (GridSearchCV 5折交叉驗證) ===
    print("正在執行決策樹 GridSearchCV 調參 (CV=5)...")
    dt_base = DecisionTreeClassifier(
        class_weight='balanced',
        random_state=42
    )
    
    # 限制 max_depth 最大為 6，避免樹過深導致過擬合與無法解釋
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 4, 5, 6],
        'min_samples_split': [10, 20, 40],
        'min_samples_leaf': [5, 10, 20]
    }
    
    grid_search = GridSearchCV(
        estimator=dt_base,
        param_grid=param_grid,
        cv=5,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    
    best_dt_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print("\n" + "=" * 60)
    print("           決策樹最佳調參參數")
    print("=" * 60)
    print(f"最佳參數組合: {best_params}")
    
    # === 6. 調參後模型評估 ===
    y_pred = best_dt_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "=" * 60)
    print("           調參後決策樹模型評估結果")
    print("=" * 60)
    print(f"測試集準確度 (Accuracy): {accuracy:.4%}")
    print("\n分類報告 (Classification Report):")
    print(classification_report(y_test, y_pred, target_names=['非長期客戶 (0)', '長期客戶 (1)']))
    print("混淆矩陣 (Confusion Matrix):")
    print(cm)
    
    # === 7. 重新繪製並保存最優決策樹結構圖 ===
    print("\n正在繪製調參後的最優決策樹結構圖...")
    
    # 計算圖表大小，根據樹的深度動態微調
    fig_width = 14 + best_params['max_depth'] * 2
    fig_height = 8 + best_params['max_depth'] * 1
    
    plt.figure(figsize=(fig_width, fig_height), dpi=150)
    annotations = plot_tree(
        best_dt_model,
        feature_names=feature_names,
        class_names=['非長期客戶', '長期客戶'],
        filled=True,
        rounded=True,
        fontsize=8,
        precision=2
    )
    
    # 四捨五入數值標籤
    value_pattern = re.compile(r'value = \[(?P<val1>[0-9.e+-]+),\s*(?P<val2>[0-9.e+-]+)\]')
    for ann in annotations:
        text = ann.get_text()
        match = value_pattern.search(text)
        if match:
            val1 = float(match.group('val1'))
            val2 = float(match.group('val2'))
            new_text = value_pattern.sub(f'value = [{int(round(val1))}, {int(round(val2))}]', text)
            ann.set_text(new_text)
            
    plt.title(f"最優決策樹長期客戶篩選路徑圖 (Depth={best_params['max_depth']}, Criterion={best_params['criterion']})", fontsize=14, fontweight='bold', pad=15)
    plt.savefig(tree_plot_path, bbox_inches='tight')
    plt.close()
    print(f"最優決策樹圖表已重新儲存至: {tree_plot_path}")

if __name__ == '__main__':
    main()
