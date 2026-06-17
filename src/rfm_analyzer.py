import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def calculate_rfm_scores(df):
    """
    計算 R_Score (近老度評分)、F_Score (消費頻率評分)、M_Score (消費金額評分) 
    以及組合的 RFM_Group 與 RFM_Score，並寫入 DataFrame 中。
    """
    df_rfm = df.copy()
    
    # 暫時填補缺失值以供評分劃分 (依據業務設定的基準)
    r_temp = df_rfm['DaySinceLastOrder'].fillna(3.0)
    f_temp = df_rfm['OrderCount'].fillna(2.0)
    
    # 1. R_Score (天數越少分數越高: 0-1天=5, 2-3天=4, 4天=3, 5-8天=2, 9天+=1)
    r_bins = [-1, 1, 3, 4, 8, np.inf]
    r_labels = [5, 4, 3, 2, 1]
    df_rfm['R_Score'] = pd.cut(r_temp, bins=r_bins, labels=r_labels).astype(int)
    
    # 2. F_Score (次數越多分數越高: 1次=1, 2次=2, 3-4次=3, 5-7次=4, 8次+=5)
    f_bins = [0, 1, 2, 4, 7, np.inf]
    f_labels = [1, 2, 3, 4, 5]
    df_rfm['F_Score'] = pd.cut(f_temp, bins=f_bins, labels=f_labels).astype(int)
    
    # 3. M_Score (回饋金越多分數越高: 使用 qcut 分成均等 5 份)
    df_rfm['M_Score'] = pd.qcut(df_rfm['CashbackAmount'], q=5, labels=[1, 2, 3, 4, 5]).astype(int)
    
    # 4. 綜合指標
    df_rfm['RFM_Group'] = df_rfm['R_Score'].astype(str) + df_rfm['F_Score'].astype(str) + df_rfm['M_Score'].astype(str)
    df_rfm['RFM_Score'] = df_rfm['R_Score'] + df_rfm['F_Score'] + df_rfm['M_Score']
    
    return df_rfm

def run_kmeans_clustering(df, n_clusters=5):
    """
    對原始 Recency, Frequency, Monetary 數值進行 Z-Score 標準化，
    並使用 K-Means 進行聚類分群，在 DataFrame 中新增 'KMeans_Cluster' 欄位。
    
    同時計算 1 到 10 群的 KMeans 內平方和 (WCSS)，用於手肘法繪圖。
    """
    df_kmeans = df.copy()
    
    # 提取原始數值並使用中位數進行暫時填補以進行聚類計算 (不改動 DataFrame 原本的 NaN)
    rfm_raw = pd.DataFrame()
    rfm_raw['Recency'] = df_kmeans['DaySinceLastOrder'].fillna(df_kmeans['DaySinceLastOrder'].median())
    rfm_raw['Frequency'] = df_kmeans['OrderCount'].fillna(df_kmeans['OrderCount'].median())
    rfm_raw['Monetary'] = df_kmeans['CashbackAmount']
    
    # Z-Score 標準化
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_raw)
    
    # 1. 尋找最佳分群數 K (肘部法則數據準備)
    wcss = []
    k_range = range(1, 11)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(rfm_scaled)
        wcss.append(km.inertia_)
        
    # 2. 執行 KMeans 聚類
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_kmeans['KMeans_Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    return df_kmeans, list(k_range), wcss
