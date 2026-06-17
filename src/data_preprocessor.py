import pandas as pd
import os
from sklearn.model_selection import train_test_split

def xlsx_to_csv(xlsx_path, csv_path):
    """
    將原始的 Excel 數據檔案轉換為 CSV 格式 (讀取 'E Comm' 工作表)
    """
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"找不到 Excel 原始檔案：{xlsx_path}")
        
    print(f"正在自 Excel 載入數據並轉存為 CSV...")
    df = pd.read_excel(xlsx_path, sheet_name='E Comm')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"轉換完成，CSV 已儲存至：'{csv_path}'")
    return df

def prepare_modeling_data(raw_csv_path, processed_csv_path, train_path, test_path):
    """
    對原始資料集進行特徵整理與 One-Hot 編碼，隨後進行 80/20 訓練與測試集分割。
    注意：此處不填補缺失值，填補統計量需在分割後依據訓練集單獨擬合以防 Data Leakage。
    """
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"找不到原始 CSV 數據：{raw_csv_path}")
        
    df = pd.read_csv(raw_csv_path)
    
    # 移除無預測意義、重複分級、以及與 KMeans 訓練結果相關的欄位
    cols_to_drop = ['CustomerID', 'RFM_Group', 'RFM_Score', 'R_Score', 'F_Score', 'M_Score', 'KMeans_Cluster']
    df_clean = df.drop(columns=cols_to_drop, errors='ignore').copy()
    print(f"已移除非預測特徵與先前訓練結果欄位：{cols_to_drop}")
    
    # 保留特徵工程：將類別變數 (object) 轉為數值獨熱編碼 (One-Hot Encoding)
    cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
    print(f"偵測到需要進行獨熱編碼的類別變數：{cat_cols}")
    
    df_processed = pd.get_dummies(df_clean, columns=cat_cols, drop_first=True, dtype=int)
    print(f"One-Hot 編碼轉換完成，特徵維度：{df_processed.shape[0]} 列, {df_processed.shape[1]} 欄")
    
    # 儲存特徵工程處理後但未填補缺失值的完整資料集
    df_processed.to_csv(processed_csv_path, index=False, encoding='utf-8-sig')
    print(f"特徵工程完整數據已儲存至：'{processed_csv_path}'")
    
    # 分割資料集 (80% 訓練集, 20% 測試集)
    train_df, test_df = train_test_split(
        df_processed, 
        test_size=0.2, 
        random_state=42, 
        stratify=df_processed['Churn']
    )
    
    # 依據使用者要求，為測試集 (test_df) 標上對應的 RFM 與聚類等分析標籤
    rfm_cols = ['CustomerID', 'R_Score', 'F_Score', 'M_Score', 'RFM_Group', 'RFM_Score', 'KMeans_Cluster']
    existing_rfm_cols = [c for c in rfm_cols if c in df.columns]
    rfm_data = df.loc[test_df.index, existing_rfm_cols]
    
    # 複製一份以避免 SettingWithCopyWarning
    test_df = test_df.copy()
    for col in reversed(existing_rfm_cols):
        test_df.insert(0, col, rfm_data[col])
    
    # 儲存分割後的資料集 (保留缺失值，測試集此時已標上 RFM 標籤)
    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')
    print(f"已將資料集劃分並存檔！(測試集已標上對應的 CustomerID 與 RFM/聚類標籤)")
    print(f"  * 訓練集：'{train_path}' ({train_df.shape[0]} 筆)")
    print(f"  * 測試集：'{test_path}' ({test_df.shape[0]} 筆)")
    
    return train_df, test_df
