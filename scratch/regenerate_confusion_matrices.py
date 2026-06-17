import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 設定畫圖美學樣式與中文字型
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Segoe UI', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

def draw_cm(cm, labels, title, filename, outputs_dir):
    plt.figure(figsize=(6.5, 5.5), dpi=150)
    
    # 繪製混淆矩陣 Heatmap (改為藍色系 Blues)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        cbar=False,
        xticklabels=labels, 
        yticklabels=labels,
        annot_kws={'size': 14, 'weight': 'bold'}
    )
    plt.title(title, fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('預測類別 (Predicted Label)', fontsize=10)
    plt.ylabel('實際類別 (True Label)', fontsize=10)
    
    plt.tight_layout()
    save_path = os.path.join(outputs_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"已生成並儲存混淆矩陣圖 [{filename}]: {save_path}")

def main():
    base_dir = r'c:\Users\user\Desktop\RFM'
    outputs_dir = os.path.join(base_dir, 'outputs')
    
    # 1. 客戶流失預測模型 (Churn)
    churn_cm = np.array([[879, 57], [28, 162]])
    churn_labels = ['未流失 (0)', '已流失 (1)']
    draw_cm(churn_cm, churn_labels, '最優隨機森林流失預測混淆矩陣', 'confusion_matrix_comparison.png', outputs_dir)
    
    # 2. 潛力長期客戶預測模型 (Long Term)
    long_term_cm = np.array([[705, 42], [66, 313]])
    long_term_labels = ['非長期 (0)', '長期客戶 (1)']
    draw_cm(long_term_cm, long_term_labels, '潛力長期客戶預測隨機森林混淆矩陣', 'long_term_confusion_matrix.png', outputs_dir)
    
    # 3. 客戶投訴行為預測模型 (Complain)
    complain_cm = np.array([[793, 12], [110, 211]])
    complain_labels = ['無投訴 (0)', '有投訴 (1)']
    draw_cm(complain_cm, complain_labels, '最優隨機森林投訴預測混淆矩陣', 'complain_confusion_matrix.png', outputs_dir)
    
    print("\n所有混淆矩陣圖表已重製完成，已全數換為藍色系 (Blues)！")

if __name__ == '__main__':
    main()
