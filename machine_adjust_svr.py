import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
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
print("正在執行高維度 Support Vector Regression 全機台效應非參數擬合...")
print(f"資料結構環境 -> 樣本數 N = {X.shape[0]}, 設備維度 p = {X.shape[1]} (p > N)")
print("="*60)

# 3. 配置 SVR 網格搜尋參數，優化調節能力
# 使用 Linear Kernel 以對應整體論文之線性去噪主軸
tuned_parameters = {
    'C': [0.1, 1, 10, 100],
    'epsilon': [0.01, 0.05, 0.1, 0.2]
}

print("正在進行 5 折交叉驗證網格搜尋 (Grid Search) 以優化超參數...")
svr_base = SVR(kernel='linear')
grid_search = GridSearchCV(svr_base, tuned_parameters, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X, y)

best_svr = grid_search.best_estimator_
print(f" SVR 參數優化完成！")
print(f"-> 最佳懲罰係數 (C): {grid_search.best_params_['C']}")
print(f"-> 最佳不敏感邊帶寬度 (Epsilon): {grid_search.best_params_['epsilon']}")
print(f"-> 全機台模型之總體變異解釋率 R2 Score: {best_svr.score(X, y):.4f}")
print("-"*60)

# 4. 計算 SVR 健壯超平面擬合之總體設備效應值
y_pred_svr = best_svr.predict(X)

# 5. 用「變數消去補償法」計算完全排除機台污染的淨殘差 (Machine-adjusted Residual)
# 公式： r_ij = Y_ij - Y_pred_ij (SVR)
df['MACHINE_ADJUSTED_RESIDUAL'] = y - y_pred_svr

# 6. 配置輸出的新資料集架構：僅保留基本識別資訊、原始值，與全新的 SVR 機器調整殘差
output_cols = ['LOT', 'WAFER', 'OBSERVATION', 'MACHINE_ADJUSTED_RESIDUAL']
output_df = df[output_cols]

# 7. 另外輸出並儲存為全新 CSV 檔案至指定之 Downloads 資料夾
output_path = r"C:\Users\user\Downloads\reduced_wafer_data_svr_adjusted.csv"
output_df.to_csv(output_path, index=False)

print(f" SVR 機器調整殘差結果已成功「另外」導出至新檔案：\n {output_path}")
print("="*60)

# 8. 終端機最終結果預覽
print("\n 【最終產出 SVR Machine-adjusted 數據結果預覽 (前 15 筆)】:")
print(output_df.head(15).to_string(index=False))
print("="*60)