import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
import joblib

def train_churn_model(train_path, test_path):
    """
    載入分割數據，以防止資訊洩漏的原則填補缺失值，
    接著訓練基準決策樹，並使用 GridSearchCV 配合 5 折交叉驗證對隨機森林進行調參。
    最後將最優模型與特徵清單持久化存檔。
    """
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("找不到劃分好的訓練集或測試集檔案")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # === 缺失值填補 (Imputation) - 嚴防資訊洩漏 ===
    # 僅使用訓練集 (train_df) 的統計量擬合填補值
    
    # 1. HourSpendOnApp 以訓練集眾數填補
    hour_mode = train_df['HourSpendOnApp'].mode()[0]
    train_df['HourSpendOnApp'] = train_df['HourSpendOnApp'].fillna(hour_mode)
    test_df['HourSpendOnApp'] = test_df['HourSpendOnApp'].fillna(hour_mode)
    
    # 2. CouponUsed 以 0 填補
    train_df['CouponUsed'] = train_df['CouponUsed'].fillna(0)
    test_df['CouponUsed'] = test_df['CouponUsed'].fillna(0)
    
    # 3. 其餘數值型欄位以訓練集中位數填補
    imputed_stats = {
        'HourSpendOnApp': f"Mode = {hour_mode}",
        'CouponUsed': "Constant = 0"
    }
    
    for col in train_df.columns:
        if col not in ['HourSpendOnApp', 'CouponUsed']:
            if pd.api.types.is_numeric_dtype(train_df[col]) and train_df[col].isnull().any():
                median_val = train_df[col].median()
                train_df[col] = train_df[col].fillna(median_val)
                test_df[col] = test_df[col].fillna(median_val)
                imputed_stats[col] = f"Median = {median_val}"
                
    # 驗證缺失值
    missing_train = train_df.isnull().sum().sum()
    missing_test = test_df.isnull().sum().sum()
    
    # === 分離特徵與標籤 ===
    # 確保預測特徵中不包含非預測欄位 (如 CustomerID 或 RFM 相關分析欄位)
    non_feature_cols = ['CustomerID', 'R_Score', 'F_Score', 'M_Score', 'RFM_Group', 'RFM_Score', 'KMeans_Cluster']
    X_train = train_df.drop(columns=['Churn'] + non_feature_cols, errors='ignore')
    y_train = train_df['Churn']
    
    # 確保測試集特徵欄位與訓練集特徵欄位完全一致 (忽略貼回的 CustomerID 與 RFM 標籤等分析欄位)
    X_test = test_df[X_train.columns]
    y_test = test_df['Churn']
    
    # === 1. 訓練基準決策樹模型 (固定 max_depth=4 作為對照組) ===
    dt_model = DecisionTreeClassifier(
        max_depth=4,
        class_weight='balanced',
        random_state=42
    )
    dt_model.fit(X_train, y_train)
    y_test_pred_dt = dt_model.predict(X_test)
    dt_acc = accuracy_score(y_test, y_test_pred_dt)
    dt_report = classification_report(y_test, y_test_pred_dt, target_names=['未流失', '已流失'])
    dt_cm = confusion_matrix(y_test, y_test_pred_dt)
    
    # === 2. 隨機森林模型：使用 GridSearchCV 進行 5 折交叉驗證調參 ===
    rf_base = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # 定義超參數網格
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [6, 8, 10],
        'min_samples_split': [10, 20],
        'min_samples_leaf': [5, 10]
    }
    
    # 使用 5 折交叉驗證，並以 F1-score 作為最佳化指標
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
    
    y_test_pred_rf = best_rf_model.predict(X_test)
    rf_acc = accuracy_score(y_test, y_test_pred_rf)
    rf_report = classification_report(y_test, y_test_pred_rf, target_names=['未流失', '已流失'])
    rf_cm = confusion_matrix(y_test, y_test_pred_rf)
    
    # === 模型持久化存檔 ===
    models_dir = r'c:\Users\user\Desktop\RFM\models'
    os.makedirs(models_dir, exist_ok=True)
    
    model_save_path = os.path.join(models_dir, 'best_churn_rf.joblib')
    features_save_path = os.path.join(models_dir, 'churn_features.joblib')
    
    joblib.dump(best_rf_model, model_save_path)
    joblib.dump(list(X_train.columns), features_save_path)
    
    print(f"\n[模型存檔成功] 最優流失預測模型已保存至: '{model_save_path}'")
    print(f"[特徵清單存檔成功] 流失模型特徵名稱已保存至: '{features_save_path}'")
    
    # === 特徵重要性計算 (以最佳隨機森林模型為主) ===
    importances = best_rf_model.feature_importances_
    feat_imp_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # 四捨五入保留 4 位小數
    feat_imp_df['Importance'] = feat_imp_df['Importance'].round(4)
    
    # === 計算 AUC 得分 ===
    from sklearn.metrics import roc_auc_score
    y_prob_dt = dt_model.predict_proba(X_test)[:, 1]
    y_prob_rf = best_rf_model.predict_proba(X_test)[:, 1]
    dt_auc = roc_auc_score(y_test, y_prob_dt)
    rf_auc = roc_auc_score(y_test, y_prob_rf)
    
    # 包裝回傳資訊
    model_stats = {
        'imputed_stats': imputed_stats,
        'missing_train': missing_train,
        'missing_test': missing_test,
        'train_shape': train_df.shape,
        'test_shape': test_df.shape,
        'best_params': best_params,
        'dt_model': dt_model,
        'dt_accuracy': dt_acc,
        'dt_report': dt_report,
        'dt_cm': dt_cm,
        'dt_auc': dt_auc,
        'rf_accuracy': rf_acc,
        'rf_report': rf_report,
        'rf_cm': rf_cm,
        'rf_auc': rf_auc
    }
    
    return best_rf_model, list(X_train.columns), feat_imp_df, model_stats
