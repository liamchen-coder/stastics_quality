import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# 1. 讀取最原始的降維特徵資料集
data_path = r"C:\Users\user\Downloads\reduced_wafer_data.csv"
df = pd.read_csv(data_path)

# 2. 準備自變數 (X：全廠 831 個機台特徵) 與 應變數 (y：原始觀測值)
y = df['OBSERVATION'].to_numpy()
X_df = df[[col for col in df.columns if col.startswith('T')]]
X = X_df.to_numpy()

print("="*60)
print("正在執行高維度 PCA-based Projection 全機台空間盲測去噪...")
print(f"原始資料結構 -> 樣本數 N = {X.shape[0]}, 設備維度 p = {X.shape[1]}")
print("="*60)

# 3. 執行主成分分析 (PCA) - 設定自動保留 85% 派工網路總變異的主成分個數
# 此步驟完全不參考 y 數據，展現非監督學習之客觀性
pca = PCA(n_components=0.85)
X_pca = pca.fit_transform(X)

n_components_chosen = X_pca.shape[1]
cumulative_variance = np.sum(pca.explained_variance_ratio_) * 100

print(f"PCA 空間轉換完成！")
print(f"-> 自動提取之主成分個數 (k): {n_components_chosen} 個")
print(f"-> 提煉出之主成分對派工網路的累積解釋變異量: {cumulative_variance:.2f}%")
print("-"*60)

# 4. 使用提取出的正交主成分特徵矩陣 (X_pca) 對 原始觀測值 (y) 進行多元線性迴歸
lr_model = LinearRegression()
lr_model.fit(X_pca, y)

print(f"主成分迴歸模型之總體變異解釋率 R2 Score: {lr_model.score(X_pca, y):.4f}")
print("-"*60)

# 5. 計算主成分群體效應的系統性擬合值
y_pred_pca = lr_model.predict(X_pca)

# 6. 用「變數消去補償法」計算排除機台主幹道污染的淨殘差 (Machine-adjusted Residual)
# 公式： r_ij = Y_ij - Y_pred_ij (PCA)
df['MACHINE_ADJUSTED_RESIDUAL'] = y - y_pred_pca

# 7. 配置輸出的新資料集架構：僅保留基本識別資訊、原始值，與全新的 PCA 機器調整殘差
output_cols = ['LOT', 'WAFER', 'OBSERVATION', 'MACHINE_ADJUSTED_RESIDUAL']
output_df = df[output_cols]

# 8. 另外輸出並儲存為全新 CSV 檔案至指定之 Downloads 資料夾
output_path = r"C:\Users\user\Downloads\reduced_wafer_data_pca_adjusted.csv"
output_df.to_csv(output_path, index=False)

print(f"PCA 機器調整殘差結果已成功「另外」導出至新檔案：\n👉 {output_path}")
print("="*60)

# 9. 終端機最終結果預覽
print("\n 【最終產出 PCA Machine-adjusted 數據結果預覽 (前 15 筆)】:")
print(output_df.head(15).to_string(index=False))
print("="*60)