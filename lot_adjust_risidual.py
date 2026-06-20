# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.linear_model import LassoCV, lasso_path

# # ==========================================
# # 1. 載入降維後的晶圓資料集（指定 Windows 本地路徑）
# # ==========================================
# # 使用 r"..." 確保 Windows 路徑的反斜線不會造成語法錯誤
# data_path = r"C:\Users\user\Downloads\reduced_wafer_data.csv"

# print("--- [步驟一] 開始讀取本地檔案 ---")
# if os.path.exists(data_path):
#     df = pd.read_csv(data_path)
#     print(f"檔案讀取成功！")
#     print(f"樣本數 (Wafer 數量): {df.shape[0]}")
#     print(f"原始欄位總數: {df.shape[1]}")
# else:
#     raise FileNotFoundError(f"找不到檔案，請確認檔案是否確實放在：{data_path}")

# # ==========================================
# # 2. 計算 Lot-adjusted Residual (批次均值調整)
# # ==========================================
# # 透過群組化計算每個 LOT 的 OBSERVATION 平均值，並相減提取殘差
# df['LOT_MEAN'] = df.groupby('LOT')['OBSERVATION'].transform('mean')
# df['LOT_ADJ_RESID'] = df['OBSERVATION'] - df['LOT_MEAN']

# print("\n--- [步驟二] 批次效應調整檢驗 (各 LOT 殘差平均值應趨近於 0) ---")
# print(df.groupby('LOT')['LOT_ADJ_RESID'].mean())

# # ==========================================
# # 3. 定義特徵矩陣 (X) 與 目標變數 (y)
# # ==========================================
# # 提取所有以 'T' 開頭的高維度機台路徑特徵
# t_cols = [c for c in df.columns if c.startswith('T')]
# X = df[t_cols].copy()
# y = df['LOT_ADJ_RESID'].values

# print(f"\n候選機台特徵 (T變數) 總數: {X.shape[1]}")

# # ==========================================
# # 4. 建立與配適 5折交叉驗證的 Lasso 模型
# # ==========================================
# print("\n--- [步驟三] 執行 5-Fold Cross-Validation Lasso... ---")
# # n_alphas=200 代表搜尋 200 個不同的懲罰力度，max_iter 確保模型完全收斂
# lasso_cv = LassoCV(cv=5, random_state=42, max_iter=30000, n_alphas=200)
# lasso_cv.fit(X, y)

# print(f"模型最優化超參數 (Optimal Alpha): {lasso_cv.alpha_:.5f}")

# # 提取非零係數（即被 Lasso 選中的關鍵核心機台）
# coefs = pd.Series(lasso_cv.coef_, index=t_cols)
# selected_machines = coefs[coefs != 0].sort_values(key=abs, ascending=False)

# print(f"\nLasso 最終篩選出的關鍵機台數量: {len(selected_machines)}")
# print("\n關鍵機台代碼與其迴歸係數 (Beta):")
# if len(selected_machines) > 0:
#     for name, coef in selected_machines.items():
#         print(f" 機台 {name}: 係數 = {coef:.6f}")
# else:
#     print(" 警告：在此懲罰力度下，未選出任何顯著機台。")

# # ==========================================
# # 5. 繪製圖表一：Lasso 交叉驗證均方誤差圖 (MSE Path)
# # ==========================================
# plt.figure(figsize=(8, 5))
# # 畫出每個 Alpha 對應的平均 MSE 曲線
# plt.plot(lasso_cv.alphas_, lasso_cv.mse_path_.mean(axis=-1), 'b-', label='Mean MSE', linewidth=2)
# # 標註最優 Alpha 的垂直切線
# plt.axvline(lasso_cv.alpha_, color='r', linestyle='--', label=f'Optimal Alpha ({lasso_cv.alpha_:.5f})')
# plt.xlabel('Alpha (Regularization Strength)', fontsize=11)
# plt.ylabel('Mean Squared Error (MSE)', fontsize=11)
# plt.title('Figure 4.1: Lasso Cross-Validation MSE Path', fontsize=12, fontweight='bold')
# plt.xscale('log')
# plt.legend()
# plt.grid(True, which="both", ls="--", alpha=0.5)
# plt.tight_layout()

# # 儲存高解析度圖片至下載資料夾，方便論文使用
# output_mse_path = r"C:\Users\user\Downloads\lasso_mse_cv.png"
# plt.savefig(output_mse_path, dpi=300)
# plt.close()
# print(f"\n[圖表儲存成功] 均方誤差路徑圖已儲存至：{output_mse_path}")

# # ==========================================
# # 6. 繪製圖表二：Lasso 係數收斂路徑圖 (Coefficient Path)
# # ==========================================
# # 計算完整的係數收斂軌跡
# alphas_path, coefs_path, _ = lasso_path(X, y, alphas=lasso_cv.alphas_)

# plt.figure(figsize=(10, 6))
# # 遍歷所有特徵，選中的用彩色加粗顯示，未選中的用灰色淡化表示
# for i in range(coefs_path.shape[0]):
#     if t_cols[i] in selected_machines.index:
#         plt.plot(alphas_path, coefs_path[i, :], label=t_cols[i], linewidth=2.5)
#     else:
#         plt.plot(alphas_path, coefs_path[i, :], color='gray', alpha=0.15, linewidth=0.5)

# plt.axvline(lasso_cv.alpha_, color='r', linestyle='--', label=f'Selected Alpha ({lasso_cv.alpha_:.5f})')
# plt.xlabel('Alpha (Regularization Strength)', fontsize=11)
# plt.ylabel('Coefficients (Beta)', fontsize=11)
# plt.title('Figure 4.2: Lasso Coefficient Convergence Path', fontsize=12, fontweight='bold')
# plt.xscale('log')
# # 將圖例擺放在右側，避免遮擋線條
# if len(selected_machines) > 0:
#     plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), title="Selected Tech")
# plt.grid(True, which="both", ls="--", alpha=0.5)
# plt.tight_layout()

# # 儲存高解析度圖片至下載資料夾
# output_coef_path = r"C:\Users\user\Downloads\lasso_coef_path.png"
# plt.savefig(output_coef_path, dpi=300)
# plt.close()
# print(f"[圖表儲存成功] 係數收斂路徑圖已儲存至：{output_coef_path}")
# print("\n--- 程式執行完畢，所有碩士報告數據與圖表已準備就緒 ---")

#1 sd

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV, Lasso, lasso_path

# ==========================================
# 1. 載入資料集
# ==========================================
data_path = r"C:\Users\user\Downloads\reduced_wafer_data.csv"

print("--- [步驟一] 開始讀取本地檔案 ---")
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    print("檔案讀取成功！")
else:
    raise FileNotFoundError(f"找不到檔案，請確認位置：{data_path}")

# 計算 Lot-adjusted Residual
df['LOT_MEAN'] = df.groupby('LOT')['OBSERVATION'].transform('mean')
df['LOT_ADJ_RESID'] = df['OBSERVATION'] - df['LOT_MEAN']

# 定義 X, y
t_cols = [c for c in df.columns if c.startswith('T')]
X = df[t_cols].copy()
y = df['LOT_ADJ_RESID'].values

# ==========================================
# 2. 執行 5折交叉驗證以計算 alpha_min 與 alpha_1se
# ==========================================
print("\n--- [步驟二] 執行 5-Fold Lasso 尋找一倍標準誤 (1-se) 邊界... ---")

# 指定合理的 Alpha 搜尋區間 (從 0.1 到 0.0005)
alphas_to_test = np.logspace(-1, -3.3, 150)
lasso_cv = LassoCV(alphas=alphas_to_test, cv=5, random_state=42, max_iter=100000, tol=1e-4)
lasso_cv.fit(X, y)

# 2.1 計算每一種 Alpha 下，5折交叉驗證的平均 MSE 與 標準誤差 (Standard Error)
mean_mse = lasso_cv.mse_path_.mean(axis=-1)
std_mse = lasso_cv.mse_path_.std(axis=-1) / np.sqrt(5) # 5折 CV 的標準誤

# 2.2 找出 alpha_min 的位置與對應的最少 MSE 數值
min_idx = np.argmin(mean_mse)
alpha_min = lasso_cv.alphas_[min_idx]
target_mse = mean_mse[min_idx] + std_mse[min_idx] # 容忍上限 = 最小 MSE + 1個標準誤

# 2.3 根據 1-se 準則，在比 alpha_min 更嚴厲(更大)的 alpha 中，找出符合容忍上限的最大 alpha
se_compatible_indices = np.where((mean_mse <= target_mse) & (lasso_cv.alphas_ >= alpha_min))[0]
if len(se_compatible_indices) > 0:
    # 選擇其中最大的 alpha 作為 alpha_1se
    alpha_1se = lasso_cv.alphas_[se_compatible_indices[0]]
else:
    alpha_1se = alpha_min

print(f"最小誤差 Alpha (alpha_min) : {alpha_min:.5f}")
print(f"一倍標準誤 Alpha (alpha_1se) : {alpha_1se:.5f} (懲罰力道更嚴格、模型更精簡)")

# ==========================================
# 3. 使用 1-se 準則的 Alpha 重新配適 Lasso 模型
# ==========================================
print("\n--- [步驟三] 使用 1-se 準則進行關鍵機台篩選 ---")
lasso_1se = Lasso(alpha=alpha_1se, max_iter=100000, tol=1e-4)
lasso_1se.fit(X, y)

coefs_1se = pd.Series(lasso_1se.coef_, index=t_cols)
selected_machines_1se = coefs_1se[coefs_1se != 0].sort_values(key=abs, ascending=False)

print(f"透過 1-se 準則篩選出的關鍵核心機台數量: {len(selected_machines_1se)}")
if len(selected_machines_1se) > 0:
    print("\n【1-se 準則核心機台與影響 Beta 係數】")
    for name, coef in selected_machines_1se.items():
        print(f"  機台 {name}: 係數 = {coef:.6f}")
else:
    print(" 提示：1-se 準則太過嚴苛將所有變數歸零，改為列出單變量殘差相關性最高的前 3 個嫌疑機台：")
    corrs = X.corrwith(pd.Series(y)).abs().sort_values(ascending=False)
    for name, val in corrs.head(3).items():
        print(f" 嫌疑機台 {name}: 相關係數絕對值 = {val:.4f}")

# ==========================================
# 4. 繪製並安全儲存兩張論文標準圖表
# ==========================================
print("\n--- [步驟四] 繪製並更新標準論文圖表 ---")

# 圖表一：MSE Path 標註 1-se 位置
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.errorbar(lasso_cv.alphas_, mean_mse, yerr=std_mse, fmt='b-', ecolor='lightblue', elinewidth=1, capsize=2, label='Mean MSE ± 1 SE')
ax.axvline(alpha_min, color='g', linestyle='--', label=f'Alpha Min ({alpha_min:.4f})')
ax.axvline(alpha_1se, color='r', linestyle='--', label=f'Alpha 1-se ({alpha_1se:.4f})')
ax.set_xlabel('Alpha (Regularization Strength)')
ax.set_ylabel('Mean Squared Error (MSE)')
ax.set_title('Figure 4.1: Lasso Cross-Validation MSE with 1-se Rule', fontweight='bold')
ax.set_xscale('log')
ax.legend()
ax.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
output_mse = r"C:\Users\user\Downloads\lasso_mse_cv.png"
fig.savefig(output_mse, dpi=300)
plt.close(fig)

# 圖表二：Coefficient Path 標註 1-se 位置
alphas_path, coefs_path, _ = lasso_path(X, y, alphas=lasso_cv.alphas_)
fig2, ax2 = plt.subplots(figsize=(8.5, 5.5))
for i in range(coefs_path.shape[0]):
    if t_cols[i] in selected_machines_1se.index:
        ax2.plot(alphas_path, coefs_path[i, :], label=t_cols[i], linewidth=2.5)
    else:
        ax2.plot(alphas_path, coefs_path[i, :], color='gray', alpha=0.08, linewidth=0.5)

ax2.axvline(alpha_min, color='g', linestyle='--', label='Alpha Min')
ax2.axvline(alpha_1se, color='r', linestyle='--', label='Alpha 1-se')
ax2.set_xlabel('Alpha')
ax2.set_ylabel('Coefficients (Beta)')
ax2.set_title('Figure 4.2: Lasso Coefficient Path with 1-se Selection', fontweight='bold')
ax2.set_xscale('log')
if len(selected_machines_1se) > 0 and len(selected_machines_1se) <= 15:
    ax2.legend(loc='upper right', bbox_to_anchor=(1.15, 1), title="1-se Selected")
ax2.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
output_coef = r"C:\Users\user\Downloads\lasso_coef_path.png"
fig2.savefig(output_coef, dpi=300)
plt.close(fig2)

print(f"1. 1-se 均方誤差圖已成功覆蓋：{output_mse}")
print(f"2. 1-se 係數收斂圖已成功覆蓋：{output_coef}")
print("\n--- 1-se 核心分析執行完畢 ---")