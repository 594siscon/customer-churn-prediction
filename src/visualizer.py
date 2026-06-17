import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
import re
import os

# 設定畫圖美學樣式與中文字型
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Segoe UI', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# 客群標籤與顏色定義 (統一管理)
colors = ['#1ab2a6', '#ff7f50', '#4a90e2', '#9b59b6', '#34495e']
cluster_labels = {
    0: 'C0: 新客/小資活躍客',
    1: 'C1: 流失警訊 Super VIP',
    2: 'C2: 中產主力忠誠客',
    3: 'C3: 流失邊緣一般客',
    4: 'C4: 活躍高金額大戶'
}

def plot_rfm_distributions(df, output_path):
    """
    繪製 R (DaySinceLastOrder)、F (OrderCount) 與 M (CashbackAmount) 的直方圖分佈
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
    fig.suptitle('RFM 欄位資料分佈分析 (RFM Feature Distributions)', fontsize=16, fontweight='bold', color='#2c3e50')
    
    # 1. R: DaySinceLastOrder (聚焦在 20 天以內以使分佈清晰)
    sns.histplot(df['DaySinceLastOrder'].dropna(), kde=True, ax=axes[0], color='#1ab2a6', bins=range(0, 22))
    axes[0].set_xlim(0, 20)
    axes[0].set_xticks(range(0, 21, 2))
    axes[0].set_title('R: DaySinceLastOrder\n(距離上次下單天數)', fontsize=12, fontweight='bold', pad=10)
    axes[0].set_xlabel('天數 (Days)', fontsize=10)
    axes[0].set_ylabel('客戶數量 (Count)', fontsize=10)
    
    # 2. F: OrderCount (countplot)
    order_counts = df['OrderCount'].dropna().astype(int)
    sns.countplot(x=order_counts, ax=axes[1], hue=order_counts, palette="Blues_d", legend=False)
    axes[1].set_title('F: OrderCount\n(上月訂單數)', fontsize=12, fontweight='bold', pad=10)
    axes[1].set_xlabel('訂單次數 (Order Count)', fontsize=10)
    axes[1].set_ylabel('客戶數量 (Count)', fontsize=10)
    
    # 3. M: CashbackAmount
    sns.histplot(df['CashbackAmount'], kde=True, ax=axes[2], color='#ff7f50', bins=20)
    axes[2].set_title('M: CashbackAmount\n(上月平均回饋金)', fontsize=12, fontweight='bold', pad=10)
    axes[2].set_xlabel('回饋金金額 (Cashback Amount)', fontsize=10)
    axes[2].set_ylabel('客戶數量 (Count)', fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"RFM 分佈直方圖已儲存至：'{output_path}'")

def plot_kmeans_elbow(k_range, wcss, output_path):
    """
    繪製手肘法圖表
    """
    plt.figure(figsize=(8, 5), dpi=120)
    plt.plot(k_range, wcss, marker='o', linestyle='--', color='#4a90e2')
    plt.title('手肘法尋找最佳 K 值 (Elbow Method for Optimal K)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('群組數量 K (Number of Clusters)', fontsize=12)
    plt.ylabel('群內平方和 WCSS (Inertia)', fontsize=12)
    plt.xticks(k_range)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"WCSS 手肘法圖表已儲存至：'{output_path}'")

def plot_kmeans_profile(df, output_path):
    """
    繪製各群組特徵偏差值 (Z-Score 柱狀圖)
    """
    rfm_raw = pd.DataFrame()
    rfm_raw['Recency'] = df['DaySinceLastOrder'].fillna(df['DaySinceLastOrder'].median())
    rfm_raw['Frequency'] = df['OrderCount'].fillna(df['OrderCount'].median())
    rfm_raw['Monetary'] = df['CashbackAmount']
    rfm_raw['Cluster'] = df['KMeans_Cluster']
    
    # 計算 Z-Score
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_raw[['Recency', 'Frequency', 'Monetary']])
    rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])
    rfm_scaled_df['Cluster'] = rfm_raw['Cluster']
    
    cluster_centers = rfm_scaled_df.groupby('Cluster').mean()
    
    # 轉為長數據
    cluster_centers_melted = cluster_centers.reset_index().melt(id_vars='Cluster', var_name='Metric', value_name='Z-Score')
    metric_translation = {'Recency': 'R: 天數偏差值', 'Frequency': 'F: 次數偏差值', 'Monetary': 'M: 金額偏差值'}
    cluster_centers_melted['Metric'] = cluster_centers_melted['Metric'].map(metric_translation)
    
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    sns.barplot(x='Cluster', y='Z-Score', hue='Metric', data=cluster_centers_melted, ax=ax, palette='Set2')
    ax.axhline(0, color='black', linewidth=1.2, linestyle='--')
    ax.set_title('各客群特徵偏離平均值程度 (Z-Score)', fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel('標準化分數 (Z-Score)', fontsize=11)
    ax.set_xlabel('群組編號 (Cluster)', fontsize=11)
    ax.set_xticks(range(5))
    ax.set_xticklabels([f'C{i}' for i in range(5)])
    ax.legend(title='RFM 項目')
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"客群特徵 Z-Score 柱狀圖已儲存至：'{output_path}'")

def plot_kmeans_3d_static(df, output_path):
    """
    繪製靜態 3D 空間分佈散點圖
    """
    # 補足缺失值以供繪圖 (不影響 DataFrame 本身)
    rfm_temp = pd.DataFrame()
    rfm_temp['Recency'] = df['DaySinceLastOrder'].fillna(df['DaySinceLastOrder'].median())
    rfm_temp['Frequency'] = df['OrderCount'].fillna(df['OrderCount'].median())
    rfm_temp['Monetary'] = df['CashbackAmount']
    rfm_temp['Cluster'] = df['KMeans_Cluster']
    
    fig = plt.figure(figsize=(9, 7), dpi=150)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    for cluster in range(5):
        cluster_data = rfm_temp[rfm_temp['Cluster'] == cluster]
        ax.scatter(cluster_data['Recency'], cluster_data['Frequency'], cluster_data['Monetary'], 
                   label=cluster_labels[cluster], c=colors[cluster], s=35, alpha=0.5)
        
    ax.set_xlim(0, 20)  # 聚焦在 20 天以內
    ax.set_xlabel('R: DaySinceLastOrder (天數)', labelpad=10)
    ax.set_ylabel('F: OrderCount (次數)', labelpad=10)
    ax.set_zlabel('M: CashbackAmount (回饋金)', labelpad=10)
    ax.set_title('K-Means (K=5) 3D 空間分佈圖\n(X軸已限縮20天內)', fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 0.95))
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"KMeans 靜態 3D 圖已儲存至：'{output_path}'")

def plot_kmeans_3d_interactive(df, output_path):
    """
    使用 Plotly 繪製可動的 3D 互動散點圖並導出 HTML 網頁
    """
    # 準備補值數據 (R補3.0, F補2.0) 限制天數在 20 天以內
    df_plot = df.copy()
    df_plot['DaySinceLastOrder'] = df_plot['DaySinceLastOrder'].fillna(3.0)
    df_plot['OrderCount'] = df_plot['OrderCount'].fillna(2.0)
    df_plot['KMeans_Label'] = df_plot['KMeans_Cluster'].map(cluster_labels)
    df_plot = df_plot[df_plot['DaySinceLastOrder'] <= 20]
    
    fig = px.scatter_3d(
        df_plot, 
        x='DaySinceLastOrder', 
        y='OrderCount', 
        z='CashbackAmount',
        color='KMeans_Label',
        hover_data={'CustomerID': True, 'DaySinceLastOrder': True, 'OrderCount': True, 'CashbackAmount': True, 'KMeans_Label': False},
        labels={
            'DaySinceLastOrder': 'R (距離上次下單天數)',
            'OrderCount': 'F (上月訂單數)',
            'CashbackAmount': 'M (上月平均回饋金)'
        },
        title="K-Means (K=5) 客群 3D 互動分佈圖 (R限制20天內)",
        color_discrete_sequence=colors
    )
    
    fig.update_traces(marker=dict(size=4, opacity=0.6))
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=50),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    fig.write_html(output_path)
    print(f"KMeans 互動式 3D HTML 已儲存至：'{output_path}'")

def plot_pairwise_scatters(df, output_dir):
    """
    繪製三張二維雙變數散點圖 (R-F, F-M, R-M) 并儲存
    """
    # 準備補值資料 (R補3.0, F補2.0)
    df_plot = df.copy()
    df_plot['DaySinceLastOrder'] = df_plot['DaySinceLastOrder'].fillna(3.0)
    df_plot['OrderCount'] = df_plot['OrderCount'].fillna(2.0)
    df_plot['KMeans_Label'] = df_plot['KMeans_Cluster'].map(cluster_labels)
    df_plot_filtered = df_plot[df_plot['DaySinceLastOrder'] <= 20]
    
    # 1. R vs F
    plt.figure(figsize=(9, 6), dpi=150)
    sns.scatterplot(data=df_plot_filtered, x='DaySinceLastOrder', y='OrderCount', hue='KMeans_Label', palette=colors, alpha=0.6, s=40)
    plt.title('R vs F: 距離上次下單天數 vs 上月訂單數 關係圖', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('R: DaySinceLastOrder (天數)', fontsize=11)
    plt.ylabel('F: OrderCount (訂單次數)', fontsize=11)
    plt.xlim(0, 20)
    plt.xticks(range(0, 21, 2))
    plt.legend(title='KMeans 客群')
    path_rf = os.path.join(output_dir, 'rfm_scatter_rf.png')
    plt.savefig(path_rf, bbox_inches='tight')
    plt.close()
    
    # 2. F vs M
    plt.figure(figsize=(9, 6), dpi=150)
    sns.scatterplot(data=df_plot_filtered, x='OrderCount', y='CashbackAmount', hue='KMeans_Label', palette=colors, alpha=0.6, s=40)
    plt.title('F vs M: 上月訂單數 vs 上月平均回饋金 關係圖', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('F: OrderCount (訂單次數)', fontsize=11)
    plt.ylabel('M: CashbackAmount (回饋金金額)', fontsize=11)
    plt.legend(title='KMeans 客群')
    path_fm = os.path.join(output_dir, 'rfm_scatter_fm.png')
    plt.savefig(path_fm, bbox_inches='tight')
    plt.close()
    
    # 3. R vs M
    plt.figure(figsize=(9, 6), dpi=150)
    sns.scatterplot(data=df_plot_filtered, x='DaySinceLastOrder', y='CashbackAmount', hue='KMeans_Label', palette=colors, alpha=0.6, s=40)
    plt.title('R vs M: 距離上次下單天數 vs 上月平均回饋金 關係圖', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('R: DaySinceLastOrder (天數)', fontsize=11)
    plt.ylabel('M: CashbackAmount (回饋金金額)', fontsize=11)
    plt.xlim(0, 20)
    plt.xticks(range(0, 21, 2))
    plt.legend(title='KMeans 客群')
    path_rm = os.path.join(output_dir, 'rfm_scatter_rm.png')
    plt.savefig(path_rm, bbox_inches='tight')
    plt.close()
    
    print(f"三張 2D 對偶關係散點圖已成功儲存於：'{output_dir}'")

def plot_feature_importances(feat_imp_df, output_path):
    """
    繪製決策樹模型的特徵重要性條形圖
    """
    plt.figure(figsize=(10, 6), dpi=150)
    sns.barplot(
        x='Importance', 
        y='Feature', 
        data=feat_imp_df[feat_imp_df['Importance'] > 0], 
        palette='viridis',
        hue='Feature',
        legend=False
    )
    plt.title('決策樹特徵重要性排序 (篩選影響流失的關鍵因素)', fontsize=14)
    plt.xlabel('重要性得分', fontsize=12)
    plt.ylabel('特徵欄位', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"特徵重要性條形圖已儲存至：'{output_path}'")

def plot_decision_tree(dt_model, feature_names, output_path):
    """
    繪製決策樹結構圖，並過濾 value=[...] 使其全部四捨五入去小數點 (轉為整數)
    """
    plt.figure(figsize=(24, 12), dpi=150)
    annotations = plot_tree(
        dt_model,
        feature_names=feature_names,
        class_names=['未流失', '已流失'],
        filled=True,
        rounded=True,
        fontsize=10,
        precision=2
    )
    
    # 使用正則表達式攔截標籤文字，將其中的 value 浮點數轉換為四捨五入後的整數
    value_pattern = re.compile(r'value = \[(?P<val1>[0-9.e+-]+),\s*(?P<val2>[0-9.e+-]+)\]')
    
    for ann in annotations:
        text = ann.get_text()
        match = value_pattern.search(text)
        if match:
            val1 = float(match.group('val1'))
            val2 = float(match.group('val2'))
            val1_int = int(round(val1))
            val2_int = int(round(val2))
            new_text = value_pattern.sub(f'value = [{val1_int}, {val2_int}]', text)
            ann.set_text(new_text)
            
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"決策樹結構圖 (Value已取整) 已儲存至：'{output_path}'")

def plot_confusion_matrices(dt_cm, rf_cm, output_path):
    """
    繪製最優隨機森林模型的混淆矩陣 Heatmap
    """
    plt.figure(figsize=(7, 6), dpi=150)
    
    # 類別標籤
    labels = ['未流失 (0)', '已流失 (1)']
    
    # 繪製隨機森林混淆矩陣 (使用綠色系)
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', cbar=False,
                xticklabels=labels, yticklabels=labels,
                annot_kws={'size': 14, 'weight': 'bold'})
    plt.title('最優隨機森林模型混淆矩陣 (Random Forest)', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('預測類別 (Predicted Label)', fontsize=11)
    plt.ylabel('實際類別 (True Label)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"最優隨機森林混淆矩陣圖已儲存至：'{output_path}'")


def plot_roc_curves(dt_model, rf_model, X_test, y_test, output_path):
    """
    計算並繪製決策樹與隨機森林模型在測試集上的 ROC 曲線對比圖 (標記 AUC 分數)
    """
    from sklearn.metrics import roc_curve, auc
    
    # 1. 預測流失機率
    y_prob_dt = dt_model.predict_proba(X_test)[:, 1]
    y_prob_rf = rf_model.predict_proba(X_test)[:, 1]
    
    # 2. 計算 FPR 與 TPR
    fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    
    # 3. 計算 AUC
    auc_dt = auc(fpr_dt, tpr_dt)
    auc_rf = auc(fpr_rf, tpr_rf)
    
    # 4. 繪製圖表
    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(fpr_rf, tpr_rf, color='#2ca02c', lw=2.5, label=f'隨機森林 (RF AUC = {auc_rf:.4f})')
    plt.plot(fpr_dt, tpr_dt, color='#1f77b4', lw=2, linestyle='--', label=f'基準決策樹 (DT AUC = {auc_dt:.4f})')
    
    # 繪製隨機分類對角基準線
    plt.plot([0, 1], [0, 1], color='#7f7f7f', lw=1.2, linestyle=':')
    
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('偽陽性率 (False Positive Rate / 1 - Specificity)', fontsize=11)
    plt.ylabel('真陽性率 (True Positive Rate / Sensitivity)', fontsize=11)
    plt.title('客戶流失預測模型 ROC 曲線對比 (ROC Curve Comparison)', fontsize=13, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"ROC 曲線對比圖已儲存至：'{output_path}'")
