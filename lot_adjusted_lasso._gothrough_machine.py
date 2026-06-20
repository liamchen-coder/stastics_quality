# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.linear_model import Lasso  # 已補上匯入，解決 NameError
# import warnings
# warnings.filterwarnings('ignore')  # 封鎖所有警告訊息

# # 設定 matplotlib 顯示中文（解決 Windows/Mac 中文亂碼問題）
# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# # 1. 讀取資料集
# data_path = r"C:\Users\user\Downloads\reduced_wafer_data_with_fitted_and_residual.csv"
# df = pd.read_csv(data_path)

# y = df['LOT_ADJUSTED_RESIDUAL']
# X = df[[col for col in df.columns if col.startswith('T')]]
# feature_names = X.columns
# X_matrix = X.to_numpy()
# total_wafers = len(df)

# # ==========================================
# # 圖一：手動調小 Alpha 之特徵釋放軌跡圖
# # ==========================================
# alpha_list = [0.10, 0.05, 0.01, 0.005, 0.002, 0.001]
# feature_counts = []

# for a in alpha_list:
#     model = Lasso(alpha=a, fit_intercept=False, max_iter=100000, random_state=42)
#     model.fit(X_matrix, y)
#     count = np.sum(model.coef_ != 0)
#     feature_counts.append(count)

# plt.figure(figsize=(9, 5.5))
# plt.plot(alpha_list, feature_counts, marker='o', color='#1f77b4', linewidth=2.5, markersize=8, label='特徵釋放數量')

# # 【核心修正】將 '★' 改成內建的 '*'，解決舊版本 Matplotlib 不支援特殊符號的問題
# plt.plot(0.05, 3, marker='*', color='#d62728', markersize=16, label='黃金臨界點 (Alpha = 0.05)')

# for i, (a, c) in enumerate(zip(alpha_list, feature_counts)):
#     plt.annotate(f'{c} 個', (a, c), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

# plt.gca().invert_xaxis()  # 反轉 X 軸
# plt.title('Lasso 模型特徵釋放路徑軌跡圖（Alpha 由高到低）', fontsize=14, fontweight='bold', pad=15)
# plt.xlabel('懲罰超參數 Alpha (向左遞減)', fontsize=12)
# plt.ylabel('篩選出之關鍵機台特徵數量', fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend(loc='upper left', fontsize=11)
# plt.tight_layout()
# plt.savefig('lasso_alpha_trajectory.png', dpi=300)
# plt.show()

# print("圖一已成功儲存至：lasso_alpha_trajectory.png")

# # ==========================================
# # 圖二：核心機台廠內晶圓覆蓋率（優化網格線最終版）
# # ==========================================
# target_machines = ['T0447', 'T1193', 'T0438']
# counts = [50, 75, 73]  
# percentages = [(c / total_wafers) * 100 for c in counts]

# fig, ax1 = plt.subplots(figsize=(8, 5))

# # 自訂三個長條的顏色（深藍、中藍、淺藍）
# bar_colors = ['#1f77b4', '#4682b4', '#6baed6']

# # 繪製長條圖
# bars = ax1.bar(target_machines, counts, color=bar_colors, width=0.5, edgecolor='black', linewidth=0.7)

# ax1.set_ylabel('行經晶圓片數 (片)', fontsize=12, color='#1f77b4')
# ax1.tick_params(axis='y', labelcolor='#1f77b4')
# ax1.set_ylim(0, 100)

# # 只允許左邊 Y 軸顯示橫向背景網格線
# ax1.grid(axis='y', linestyle='--', alpha=0.5, color='#cccccc')

# # 建立右軸：廠內覆蓋率 (百分比)
# ax2 = ax1.twinx()
# ax2.set_ylabel('廠內覆蓋率 (%)', fontsize=12, color='#d62728')
# ax2.tick_params(axis='y', labelcolor='#d62728')
# ax2.set_ylim(0, 40)

# # 明確關閉右軸的網格線
# ax2.grid(False)

# # 在長條圖上方加上數值與百分比標籤
# for i, bar in enumerate(bars):
#     height = bar.get_height()
#     ax1.text(bar.get_x() + bar.get_width()/2., height + 3,
#              f'{counts[i]} 片\n({percentages[i]:.2f}%)',
#              ha="center", va="bottom", fontsize=11, fontweight='bold')

# plt.title('核心異常機台之廠內晶圓流量與覆蓋率分析表 ($\alpha = 0.05$)', fontsize=14, fontweight='bold', pad=15)
# ax1.set_xlabel('關鍵機台代碼', fontsize=12)
# plt.tight_layout()
# plt.savefig('machine_wafer_coverage.png', dpi=300)
# plt.show()

# print("圖二已成功更新並儲存至：machine_wafer_coverage.png")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker  # 👈 引入刻度控制模組，用來強制整數刻度
import warnings
warnings.filterwarnings('ignore')  # 封鎖所有無謂的警告訊息

# 設定 matplotlib 顯示中文（解決 Windows/Mac 中文亂碼問題）
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 讀取降維特徵資料集
data_path = r"C:\Users\user\Downloads\reduced_wafer_data.csv"
df = pd.read_csv(data_path)

# 2. 定義要統計的關鍵機台欄位
target_machines = ['T0447', 'T1193', 'T0438']

# 依據 LOT 進行分組加總
lot_summary = df.groupby('LOT')[target_machines].sum().astype(int)

# 定義標準的 1 到 10 排列順序，強迫 DataFrame 依照此順序重新排序
ordered_lots = ['LOT1', 'LOT2', 'LOT3', 'LOT4', 'LOT5', 'LOT6', 'LOT7', 'LOT8', 'LOT9', 'LOT10']
lot_summary = lot_summary.reindex(ordered_lots)

print("="*60)
print("【各批次 (LOT 1~10 排序) 之關鍵機台晶圓流量統計表】")
print("="*60)
print(lot_summary)
print("="*60)

# 3. 繪製精美分組長條圖
fig, ax = plt.subplots(figsize=(13, 6))

# 使用明確的深藍、中藍、淺藍色系繪製分組圖
lot_summary.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', linewidth=0.7,
                 color=['#1f77b4', '#4682b4', '#6baed6'])

# 設定圖表標題與軸標籤
ax.set_title('各生產批次 (LOT 1~10 順序) 行經關鍵異常機台之晶圓數量分佈圖', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('生產批次名稱 (LOT)', fontsize=12)
ax.set_ylabel('行經晶圓片數 (片)', fontsize=12)

# 微調 X 軸標籤角度，讓字體水平呈現更易閱讀
plt.xticks(rotation=0)

# 微調 Y 軸範圍，留出空間給上方的數字標籤
ax.set_ylim(0, 20)

# 【核心修正】強迫 Y 軸的刻度（Ticks）只能是整數，消滅所有小數點
ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

# 設定乾淨、淡淡的灰色橫向背景網格線
ax.grid(axis='y', linestyle='--', alpha=0.5, color='#cccccc')

# 在每一個獨立的長條圖正上方，精確加上晶圓片數的文字標籤
for p in ax.patches:
    height = p.get_height()
    if height > 0:  # 只有當片數大於 0 時才顯示，保持畫面乾淨
        ax.annotate(f'{int(height)}', 
                    (p.get_x() + p.get_width() / 2., height), 
                    ha='center', va='bottom', fontsize=9, fontweight='bold', 
                    textcoords='offset points', xytext=(0, 3))

# 調整排版防止標籤被切到
plt.tight_layout()

# 儲存高品質圖檔
output_fig_path = 'lot_machine_distribution_ordered.png'
plt.savefig(output_fig_path, dpi=300)
plt.show()

print(f"排序+整數刻度版各 LOT 分佈長條圖已成功繪製並儲存至：{output_fig_path}")