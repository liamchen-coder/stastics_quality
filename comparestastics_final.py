import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 1. 讀取最原始的降維特徵資料集
data_path = r"C:\Users\user\Downloads\reduced_wafer_data.csv"
df = pd.read_csv(data_path)

# ====================================================================
# 🛑 請在這裡填入你第四章抓出來的最毒、最異常的那個機台代號！
# 假設是 T0447，請自行替換成你報告中的真凶
TARGET_MACHINE = 'T0086'  # <--- 修改這裡！
# ====================================================================

if TARGET_MACHINE not in df.columns:
    print(f"❌ 找不到機台 {TARGET_MACHINE}，請確認名稱是否正確！")
else:
    print(f"🚀 啟動 6.1 綜合探索：剖析首要戰犯機台 【 {TARGET_MACHINE} 】 的破壞力")

    # 2. 將晶圓分為「經過該機台(1)」與「未經過該機台(0)」兩大群體
    df['Machine_Exposure'] = df[TARGET_MACHINE].map({1: f'Exposed to {TARGET_MACHINE}', 0: 'Not Exposed (Healthy)'})
    
    # 計算兩群體的平均值與偏移量
    mean_exposed = df[df[TARGET_MACHINE] == 1]['OBSERVATION'].mean()
    mean_healthy = df[df[TARGET_MACHINE] == 0]['OBSERVATION'].mean()
    shift_amount = mean_exposed - mean_healthy
    
    # 執行 T 檢定確認顯著性
    t_stat, p_val = stats.ttest_ind(
        df[df[TARGET_MACHINE] == 1]['OBSERVATION'],
        df[df[TARGET_MACHINE] == 0]['OBSERVATION'],
        equal_var=False
    )
    
    print(f"\n📊 【 {TARGET_MACHINE} 工程數據量化報告 】")
    print(f"  -> 經過該機台的晶圓平均值: {mean_exposed:.4f}")
    print(f"  -> 未經過的晶圓基準平均值: {mean_healthy:.4f}")
    print(f"  -> 產生之系統性物理偏移量: {shift_amount:.4f}")
    print(f"  -> Welch's T-test 顯著性: p-value = {p_val:.4e}")
    
    if p_val < 0.05:
        print(f"  -> 📢 結論：極度顯著！只要晶圓流經 {TARGET_MACHINE}，必然引發嚴重的觀測值基準線偏誤！")
        
    # ====================================================================
    # 🎨 3. 繪製學術級：小提琴與箱型圖疊加視覺化 (Violin + Boxplot)
    # ====================================================================
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(10, 6))
    
    # 畫出小提琴圖 (展示數據的分佈形狀)
    ax = sns.violinplot(
        x="Machine_Exposure", y="OBSERVATION", data=df, 
        inner=None, color=".8", linewidth=0, alpha=0.5
    )
    
    # 畫出箱型圖 (展示中位數與四分位距 IQR)
    sns.boxplot(
        x="Machine_Exposure", y="OBSERVATION", data=df, 
        width=0.3, boxprops={'zorder': 2}, ax=ax
    )
    
    # 畫出個別晶圓的散佈點 (Swarmplot)，增加視覺豐富度
    sns.stripplot(
        x="Machine_Exposure", y="OBSERVATION", data=df, 
        size=4, color="black", alpha=0.3, ax=ax, jitter=True
    )
    
    # 美化圖表與標籤
    plt.title(f"Comprehensive Exploration of Key Machine ({TARGET_MACHINE})\nSystematic Shift in Process Baseline", fontsize=15, fontweight='bold', pad=15)
    plt.xlabel("Machine Exposure Status", fontsize=12, fontweight='bold')
    plt.ylabel("Original OBSERVATION Value", fontsize=12, fontweight='bold')
    
    # 加上 P-value 與偏移量的文字標籤
    plt.text(0.5, df['OBSERVATION'].max() * 0.95, 
             f"Shift = {shift_amount:.3f}\np-value = {p_val:.2e}", 
             horizontalalignment='center', fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))
    
    plt.tight_layout()
    
    # 另存圖檔到下載資料夾
    output_image_path = r"C:\Users\user\Downloads\Key_Machine_Exploration_Plot.png"
    plt.savefig(output_image_path, dpi=300)
    print(f"\n💾 學術級對比視覺化圖表已導出至：{output_image_path}")
    
    plt.show()