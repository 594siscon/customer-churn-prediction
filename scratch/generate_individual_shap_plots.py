import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from sklearn.model_selection import train_test_split

# 設定畫圖美學樣式與中文字型
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Segoe UI', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def generate_churn_shap_plots(base_dir, outputs_dir, models_dir):
    print("\n--- [1/3] 正在處理客戶流失預測模型 (Churn) ---")
    train_path = os.path.join(base_dir, 'data', 'train_set.csv')
    test_path = os.path.join(base_dir, 'data', 'test_set.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("錯誤: 找不到流失預測訓練集或測試集。")
        return
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # 缺失值填補 (訓練集統計量)
    hour_mode = train_df['HourSpendOnApp'].mode()[0]
    train_df['HourSpendOnApp'] = train_df['HourSpendOnApp'].fillna(hour_mode)
    test_df['HourSpendOnApp'] = test_df['HourSpendOnApp'].fillna(hour_mode)
    
    train_df['CouponUsed'] = train_df['CouponUsed'].fillna(0)
    test_df['CouponUsed'] = test_df['CouponUsed'].fillna(0)
    
    for col in train_df.columns:
        if col not in ['HourSpendOnApp', 'CouponUsed']:
            if pd.api.types.is_numeric_dtype(train_df[col]) and train_df[col].isnull().any():
                median_val = train_df[col].median()
                train_df[col] = train_df[col].fillna(median_val)
                test_df[col] = test_df[col].fillna(median_val)
                
    model_path = os.path.join(models_dir, 'best_churn_rf.joblib')
    features_path = os.path.join(models_dir, 'churn_features.joblib')
    
    rf_model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    X_test = test_df[feature_names]
    
    explainer = shap.TreeExplainer(rf_model)
    explanation = explainer(X_test)
    explanation_class1 = explanation[:, :, 1] if len(explanation.shape) == 3 else explanation
    
    # 繪製各特徵圖
    target_feats = {
        'Tenure': ('churn_shap_tenure.png', '客戶年資 (Tenure) 相依性'),
        'Complain': ('churn_shap_complain.png', '客戶投訴 (Complain) 相依性'),
        'MaritalStatus_Single': ('churn_shap_marital_single.png', '婚姻狀況：單身 (MaritalStatus_Single) 相依性'),
        'CityTier': ('churn_shap_citytier.png', '居住城市等級 (CityTier) 相依性')
    }
    
    for feat, (filename, title) in target_feats.items():
        plt.figure(figsize=(7, 4.8), dpi=150)
        shap.plots.scatter(explanation_class1[:, feat], color=explanation_class1[:, feat], show=False)
        plt.title(title, fontsize=12, fontweight='bold', pad=10)
        plt.ylabel("SHAP 值 (對流失機率的邊際貢獻)", fontsize=10)
        plt.axhline(0, color='red', linestyle='--', alpha=0.6)
        plt.tight_layout()
        save_path = os.path.join(outputs_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"已儲存單圖: {save_path}")

def generate_long_term_shap_plots(base_dir, outputs_dir, models_dir):
    print("\n--- [2/3] 正在處理潛力長期客戶預測模型 (Long Term) ---")
    processed_csv_path = os.path.join(base_dir, 'data', 'E Commerce Dataset Processed.csv')
    
    if not os.path.exists(processed_csv_path):
        print(f"錯誤: 找不到處理後的資料集: {processed_csv_path}")
        return
        
    df = pd.read_csv(processed_csv_path)
    df['Is_Long_Term'] = (df['Tenure'] > 12.0).astype(int)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Is_Long_Term'])
    
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
                
    model_path = os.path.join(models_dir, 'best_long_term_rf.joblib')
    features_path = os.path.join(models_dir, 'long_term_features.joblib')
    
    rf_model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    X_test = test_df[feature_names]
    
    explainer = shap.TreeExplainer(rf_model)
    explanation = explainer(X_test)
    explanation_class1 = explanation[:, :, 1] if len(explanation.shape) == 3 else explanation
    
    # 繪製 4 大基本特徵獨立圖
    target_feats = {
        'CashbackAmount': ('long_term_shap_cashback.png', '回饋金金額 (CashbackAmount) 相依性'),
        'NumberOfAddress': ('long_term_shap_address.png', '登記地址數 (NumberOfAddress) 相依性'),
        'CityTier': ('long_term_shap_citytier.png', '居住城市等級 (CityTier) 相依性'),
        'OrderCount': ('long_term_shap_ordercount.png', '上月下單次數 (OrderCount) 相依性')
    }
    
    for feat, (filename, title) in target_feats.items():
        plt.figure(figsize=(7, 4.8), dpi=150)
        shap.plots.scatter(explanation_class1[:, feat], color=explanation_class1[:, feat], show=False)
        plt.title(title, fontsize=12, fontweight='bold', pad=10)
        plt.ylabel("SHAP 值 (對長期機率的邊際貢獻)", fontsize=10)
        plt.axhline(0, color='red', linestyle='--', alpha=0.6)
        plt.tight_layout()
        save_path = os.path.join(outputs_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"已儲存單圖: {save_path}")
        
    # 繪製 4 大品類特徵獨立圖
    target_cats = {
        'PreferedOrderCat_Mobile': ('long_term_shap_cat_mobile.png', '偏好手機類別 (PreferedOrderCat_Mobile) 相依性'),
        'PreferedOrderCat_Mobile Phone': ('long_term_shap_cat_mobilephone.png', '偏好行動電話類別 (PreferedOrderCat_Mobile Phone) 相依性'),
        'PreferedOrderCat_Grocery': ('long_term_shap_cat_grocery.png', '偏好雜貨類別 (PreferedOrderCat_Grocery) 相依性'),
        'PreferedOrderCat_Others': ('long_term_shap_cat_others.png', '偏好其他類別 (PreferedOrderCat_Others) 相依性')
    }
    
    for feat, (filename, title) in target_cats.items():
        plt.figure(figsize=(7, 4.8), dpi=150)
        shap.plots.scatter(explanation_class1[:, feat], color=explanation_class1[:, feat], show=False)
        plt.title(title, fontsize=12, fontweight='bold', pad=10)
        plt.ylabel("SHAP 值 (對長期機率的邊際貢獻)", fontsize=10)
        plt.axhline(0, color='red', linestyle='--', alpha=0.6)
        plt.tight_layout()
        save_path = os.path.join(outputs_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"已儲存單圖: {save_path}")

def generate_complain_shap_plots(base_dir, outputs_dir, models_dir):
    print("\n--- [3/3] 正在處理客戶投訴行為預測模型 (Complain) ---")
    processed_csv_path = os.path.join(base_dir, 'data', 'E Commerce Dataset Processed.csv')
    
    if not os.path.exists(processed_csv_path):
        print(f"錯誤: 找不到處理後的資料集: {processed_csv_path}")
        return
        
    df = pd.read_csv(processed_csv_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Complain'])
    
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
                
    model_path = os.path.join(models_dir, 'best_complain_rf.joblib')
    features_path = os.path.join(models_dir, 'complain_features.joblib')
    
    rf_model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    X_test = test_df[feature_names]
    
    explainer = shap.TreeExplainer(rf_model)
    explanation = explainer(X_test)
    explanation_class1 = explanation[:, :, 1] if len(explanation.shape) == 3 else explanation
    
    # 繪製 3 大核心特徵獨立圖
    target_feats = {
        'Tenure': ('complain_shap_tenure.png', '客戶年資 (Tenure) 相依性'),
        'WarehouseToHome': ('complain_shap_warehouse.png', '配送距離 (WarehouseToHome) 相依性'),
        'Gender_Male': ('complain_shap_gender.png', '性別：男性 (Gender_Male) 相依性')
    }
    
    for feat, (filename, title) in target_feats.items():
        plt.figure(figsize=(7, 4.8), dpi=150)
        
        if feat == 'WarehouseToHome':
            # 排除極端異常點 (>120) 以避免 X 軸過度延展
            mask = (X_test['WarehouseToHome'] <= 120).values
            shap.plots.scatter(explanation_class1[mask, feat], color=explanation_class1[mask, feat], show=False)
        else:
            shap.plots.scatter(explanation_class1[:, feat], color=explanation_class1[:, feat], show=False)
            
        plt.title(title, fontsize=12, fontweight='bold', pad=10)
        plt.ylabel("SHAP 值 (對投訴機率的邊際貢獻)", fontsize=10)
        plt.axhline(0, color='red', linestyle='--', alpha=0.6)
        plt.tight_layout()
        save_path = os.path.join(outputs_dir, filename)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        print(f"已儲存單圖: {save_path}")

def main():
    base_dir = r'c:\Users\user\Desktop\RFM'
    outputs_dir = os.path.join(base_dir, 'outputs')
    models_dir = os.path.join(base_dir, 'models')
    
    print("開始為所有模型特徵產生獨立小圖...")
    
    # 1. Churn
    generate_churn_shap_plots(base_dir, outputs_dir, models_dir)
    
    # 2. Long Term
    generate_long_term_shap_plots(base_dir, outputs_dir, models_dir)
    
    # 3. Complain
    generate_complain_shap_plots(base_dir, outputs_dir, models_dir)
    
    print("\n所有獨立小圖皆已成功產生並儲存！")

if __name__ == '__main__':
    main()
