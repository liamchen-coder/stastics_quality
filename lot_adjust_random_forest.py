import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings('ignore')  # 封鎖所有無謂的警告訊息

# 設定 matplotlib 顯示中文（解決 Windows/Mac 中文亂碼問題）
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 讀取降維特徵殘差資料集
data_path = r"C:\Users\user\Downloads\reduced_wafer_data_with_fitted_and_residual.csv"
df = pd.read_csv(data_path)

# 2. 準備自變數 (X) 與應變數 (y)
y = df['LOT_ADJUSTED_RESIDUAL'].to_numpy()
X_df = df[[col for col in df.columns if col.startswith('T')]]
X = X_df.to_numpy()
feature_names = list(X_df.columns)

print("="*60)
print("正在執行隨機森林 (Random Forest) 特徵重要性分析...")
print("="*60)

# 3. 建立並擬合隨機森林迴歸模型
# n_estimators=500 可確保小樣本下的重要性得分具備高度穩健與收斂性
# n_jobs=-1 開啟多核心並行加速
rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
rf.fit(X, y)

# 4. 提取特徵重要性 (Feature Importance) 並進行降序排序
importances = rf.feature_importances_

importance_df = pd.DataFrame({
    '機台特徵': feature_names,
    '特徵重要性 (Importance)': importances
}).sort_values(by='特徵重要性 (Importance)', ascending=False).reset_index(drop=True)

# 5. 終端機文字輸出：前 10 大核心關鍵機台
print("🎯 【隨機森林 - 前 10 大核心關鍵機台特徵重要性排行】")
print("-"*60)
print(importance_df.head(10).to_string(index=True))
print("="*60)

# 6. 繪製精美特徵重要性長條圖 (嚴格依據得分由高到低排序)
top_n = 15  # 繪製前 15 大最具影響力的機台
plt.figure(figsize=(12, 6))

# 使用質感深藍色系繪製
plt.bar(importance_df['機台特徵'].head(top_n), importance_df['特徵重要性 (Importance)'].head(top_n),
        color='#2b5c8f', edgecolor='black', linewidth=0.7, width=0.6)

# 設定學術規格之圖表標題與軸標籤
plt.title(f'隨機森林模型 - 前 {top_n} 大核心關鍵機台特徵重要性分佈圖 (Lot-adjusted)', fontsize=14, fontweight='bold', pad=15)
plt.xlabel('關鍵機台代碼', fontsize=12)
plt.ylabel('特徵重要性得分 (Gini Importance)', fontsize=12)

# 微調 X 軸標籤角度防止重疊
plt.xticks(rotation=45)

# 視覺優化：橫向背景網格線
plt.grid(axis='y', linestyle='--', alpha=0.5, color='#cccccc')
plt.tight_layout()

# 儲存高品質圖檔 (300 DPI)
output_fig_path = 'rf_feature_importance.png'
plt.savefig(output_fig_path, dpi=300)
plt.show()

print(f"📈 隨機森林特徵重要性長條圖已成功繪製並儲存至：{output_fig_path}")