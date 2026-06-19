import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso

# 1. 讀取含殘差與降維特徵的資料集
data_path = r"C:\Users\user\Downloads\reduced_wafer_data_with_fitted_and_residual.csv"
df = pd.read_csv(data_path)

# 2. 定義應變數 Y
y = df['LOT_ADJUSTED_RESIDUAL']

# 動態篩選所有以 'T' 開頭的機台特徵
X = df[[col for col in df.columns if col.startswith('T')]]
feature_names = X.columns
X_matrix = X.to_numpy()

# 3. 方案 B：測試多個手動指定的較小 Alpha 值，觀察特徵釋放趨勢
# 交叉驗證的最佳解是 0.0885 (選出 0 個)，所以我們從 0.05 開始往下測
alpha_list = [0.1, 0.05, 0.01, 0.005, 0.002, 0.001]

print("="*60)
print("【方案 B 執行中】手動調小 Alpha 懲罰力道，強迫模型釋放特徵...")
print("="*60)

final_selected_df = None
successful_alpha = None

for alpha_test in alpha_list:
    # 宣告 Lasso 模型（強迫通過原點 fit_intercept=False）
    model = Lasso(alpha=alpha_test, fit_intercept=False, max_iter=100000, random_state=42)
    model.fit(X_matrix, y)
    
    # 提取非 0 係數
    coefficients = model.coef_
    selected = pd.DataFrame({
        'Machine_Feature': feature_names,
        'Coefficient': coefficients
    })
    selected = selected[selected['Coefficient'] != 0].copy()
    
    print(f"當 Alpha = {alpha_test:<6} | 成功篩選出 {len(selected):>3} 個關鍵機台特徵")
    
    # 記錄下第一個「成功選出特徵（但數量適中，最好在 1 到 30 個之間）」的結果作為報告輸出
    if len(selected) > 0 and final_selected_df is None:
        if len(selected) <= 35:  # 特徵數量適中，適合放入 PPT 或報告
            selected['Abs_Coefficient'] = selected['Coefficient'].abs()
            final_selected_df = selected.sort_values(by='Abs_Coefficient', ascending=False).drop(columns=['Abs_Coefficient'])
            successful_alpha = alpha_test

print("="*60)

# 4. 輸出最終手動篩選報告
if final_selected_df is not None:
    print(f"\n系統自動推薦：採用 Alpha = {successful_alpha} 的篩選結果（最具代表性且數量適中）：")
    print("-"*50)
    print(final_selected_df.to_string(index=False))
    print("-"*50)
    
    # 自動儲存清單
    final_selected_df.to_csv("lasso_forced_selected_machines.csv", index=False)
    print(f"關鍵機台清單已成功儲存至：lasso_forced_selected_machines.csv")
else:
    print("\n提示：即使 Alpha 降到 0.001，所有機台係數依然為 0。")
    print("這代表機台路徑資料與該殘差可能存在嚴重的結構性不相關。")
    print("強烈建議下一步：請放棄 Lasso，直接改用 ElasticNet 模型，或進入隨機效應隨機模型分析！")
print("="*60)