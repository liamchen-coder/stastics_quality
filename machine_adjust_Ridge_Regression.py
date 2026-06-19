import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
import warnings
warnings.filterwarnings('ignore')  # 封鎖高維度下無謂的統計警告

# 1. 讀取最原始的降維特徵資料集（請確保執行時，原始檔案在同一個目錄下）
data_path = r"C:\Users\user\Downloads\reduced_wafer_data.csv"
df = pd.read_csv(data_path)

# 2. 準備自變數 (X：全廠 831 個機台特徵) 與 應變數 (y：原始觀測值)
y = df['OBSERVATION'].to_numpy()
X_df = df[[col for col in df.columns if col.startswith('T')]]
X = X_df.to_numpy()

print("="*60)
print("正在執行高維度 Ridge Regression 全機台效應集體擬合...")
print(f"資料結構環境 -> 樣本數 N = {X.shape[0]}, 設備維度 p = {X.shape[1]} (p > N)")
print("="*60)

# 3. 使用 RidgeCV (內建 5 折交叉驗證) 自動尋找最佳的懲罰參數 alpha (lambda)
alphas_to_test = np.logspace(-3, 5, 200)
ridge_model = RidgeCV(alphas=alphas_to_test, cv=5, scoring='neg_mean_squared_error')
ridge_model.fit(X, y)

print(f"優化完成！模型選擇的最佳正則化懲罰參數 Alpha: {ridge_model.alpha_:.4f}")
print(f"全機台模型之總體變異解釋率 R2 Score: {ridge_model.score(X, y):.4f}")
print("-"*60)

# 4. 計算全機台群體效應的擬合值 (f(X))
y_pred_ridge = ridge_model.predict(X)

# 5. 用「變數消去補償法」計算完全排除機台污染的淨殘差 (Machine-adjusted Residual)
# 公式： r_ij = Y_ij - Y_pred_ij
df['MACHINE_ADJUSTED_RESIDUAL'] = y - y_pred_ridge

# 6. 配置輸出的新資料集架構：保留基本識別資訊、原始值，與全新的機器調整殘差
output_cols = ['LOT', 'WAFER', 'OBSERVATION', 'MACHINE_ADJUSTED_RESIDUAL']
output_df = df[output_cols]

# 7. 另外輸出並儲存為全新 CSV 檔案，直接指定路徑到你的 Downloads 資料夾
output_path = r"C:\Users\user\Downloads\reduced_wafer_data_ridge_adjusted.csv"
output_df.to_csv(output_path, index=False)

print(f"💾 機器調整殘差結果已成功「另外」導出至新檔案：\n👉 {output_path}")
print("="*60)

# 8. 終端機最終結果預覽
print("\n🎯 【最終產出 Machine-adjusted 數據結果預覽 (前 15 筆)】:")
print(output_df.head(15).to_string(index=False))
print("="*60)