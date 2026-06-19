# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import warnings
# warnings.filterwarnings('ignore')

# # ==========================================
# # 1. 讀取資料與設定
# # ==========================================
# # 請確認這裡的路徑與您電腦中的檔案路徑一致
# data_path = r"C:\Users\user\Downloads\reduced_wafer_data.csv"
# df = pd.read_csv(data_path)

# # 設定要觀察的最新三大關鍵異常機台
# target_machines = ['T0447', 'T0872', 'T1202']

# # ==========================================
# # 2. 資料分組與排序處理
# # ==========================================
# # 計算各 LOT 經過這些機台的總片數 (假設經過為 1，未經過為 0)
# lot_counts = df.groupby('LOT')[target_machines].sum()

# # 為了確保 X 軸是完美的 LOT1 ~ LOT10 順序，我們自訂排序規則
# def extract_lot_num(lot_str):
#     try:
#         return int(lot_str.replace('LOT', ''))
#     except:
#         return 0

# # 將索引依照 LOT 後面的數字大小進行排序
# sorted_idx = lot_counts.index.map(extract_lot_num).argsort()
# lot_counts = lot_counts.iloc[sorted_idx]

# # ==========================================
# # 3. 開始繪製精美圖表
# # ==========================================
# # 設定中文字體 (Windows 預設微軟正黑體)
# plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
# plt.rcParams['axes.unicode_minus'] = False

# fig, ax = plt.subplots(figsize=(15, 6))

# x = np.arange(len(lot_counts.index))  # X 軸的 LOT 座標點
# width = 0.25  # 每個長條圖的寬度

# # 定義具有層次感的三種藍色 (呼應您原圖的配色風格)
# colors = ['#1f77b4', '#4682B4', '#7CB9E8']

# # 繪製三台機台的長條圖 (利用平移 x 座標達到並排效果)
# rects1 = ax.bar(x - width, lot_counts['T0447'], width, label='T0447', color=colors[0], edgecolor='black', linewidth=0.7)
# rects2 = ax.bar(x,         lot_counts['T0872'], width, label='T0872', color=colors[1], edgecolor='black', linewidth=0.7)
# rects3 = ax.bar(x + width, lot_counts['T1202'], width, label='T1202', color=colors[2], edgecolor='black', linewidth=0.7)

# # ==========================================
# # 4. 標籤、標題與格線美化
# # ==========================================
# ax.set_ylabel('行經晶圓片數 (片)', fontsize=12, fontweight='bold')
# ax.set_xlabel('生產批次名稱 (LOT)', fontsize=12, fontweight='bold')
# ax.set_title('各生產批次 (LOT 1~10 順序) 行經關鍵異常機台之晶圓數量分佈圖', fontsize=15, fontweight='bold')
# ax.set_xticks(x)
# ax.set_xticklabels(lot_counts.index, fontsize=11)
# ax.set_ylim(0, 20)  # Y軸上限設為 20 片
# ax.legend(fontsize=11)

# # 加入水平虛線網格線，方便對齊數值
# ax.yaxis.grid(True, linestyle='--', alpha=0.6)
# ax.set_axisbelow(True) # 讓格線藏在長條圖後方

# # ==========================================
# # 5. 在長條圖上方加入數值標籤 (值為 0 則不顯示)
# # ==========================================
# def autolabel(rects):
#     for rect in rects:
#         height = rect.get_height()
#         if height > 0:
#             ax.annotate(f'{int(height)}',
#                         xy=(rect.get_x() + rect.get_width() / 2, height),
#                         xytext=(0, 3),  # 垂直往上偏 3 像素
#                         textcoords="offset points",
#                         ha='center', va='bottom', fontsize=10, fontweight='bold')

# autolabel(rects1)
# autolabel(rects2)
# autolabel(rects3)

# # ==========================================
# # 6. 顯示並儲存圖表
# # ==========================================
# plt.tight_layout()

# # 自動將圖片存到您的下載資料夾中
# output_image_path = r"C:\Users\user\Downloads\New_Machine_Distribution.png"
# plt.savefig(output_image_path, dpi=300)
# print(f"✅ 圖表繪製完成！已高畫質儲存至：{output_image_path}")

# plt.show()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 讀取資料與設定
# ==========================================
# 請確認這裡的路徑與您電腦中的檔案路徑一致
data_path = r"C:\Users\user\Downloads\reduced_wafer_data.csv"
df = pd.read_csv(data_path)

# 設定要觀察的隨機森林前五大機台
target_machines = ['T0447', 'T2145', 'T1241', 'T1308', 'T1202']

# ==========================================
# 2. 資料分組與排序處理
# ==========================================
# 計算各 LOT 經過這些機台的總片數 (假設經過為 1，未經過為 0)
lot_counts = df.groupby('LOT')[target_machines].sum()

# 為了確保 X 軸是完美的 LOT1 ~ LOT10 順序，我們自訂排序規則
def extract_lot_num(lot_str):
    try:
        return int(lot_str.replace('LOT', ''))
    except:
        return 0

# 將索引依照 LOT 後面的數字大小進行排序
sorted_idx = lot_counts.index.map(extract_lot_num).argsort()
lot_counts = lot_counts.iloc[sorted_idx]

# ==========================================
# 3. 開始繪製精美圖表
# ==========================================
# 設定中文字體 (Windows 預設微軟正黑體)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(16, 7))

x = np.arange(len(lot_counts.index))  # X 軸的 LOT 座標點
width = 0.15  # 因為有 5 個機台，寬度要稍微縮小，讓它們可以並排

# 定義 5 種清晰易辨識的顏色
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# 繪製 5 台機台的長條圖 (利用平移 x 座標達到並排效果)
rects1 = ax.bar(x - 2*width, lot_counts['T0447'], width, label='T0447', color=colors[0], edgecolor='black', linewidth=0.7)
rects2 = ax.bar(x - width,   lot_counts['T2145'], width, label='T2145', color=colors[1], edgecolor='black', linewidth=0.7)
rects3 = ax.bar(x,           lot_counts['T1241'], width, label='T1241', color=colors[2], edgecolor='black', linewidth=0.7)
rects4 = ax.bar(x + width,   lot_counts['T1308'], width, label='T1308', color=colors[3], edgecolor='black', linewidth=0.7)
rects5 = ax.bar(x + 2*width, lot_counts['T1202'], width, label='T1202', color=colors[4], edgecolor='black', linewidth=0.7)

# ==========================================
# 4. 標籤、標題與格線美化
# ==========================================
ax.set_ylabel('行經晶圓片數 (片)', fontsize=12, fontweight='bold')
ax.set_xlabel('生產批次名稱 (LOT)', fontsize=12, fontweight='bold')
ax.set_title('各生產批次 (LOT 1~10 順序) 行經隨機森林前五大機台之晶圓數量分佈圖', fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(lot_counts.index, fontsize=11)
ax.set_ylim(0, 20)  # Y軸上限設為 20 片
ax.legend(fontsize=11, loc='upper right')

# 加入水平虛線網格線，方便對齊數值
ax.yaxis.grid(True, linestyle='--', alpha=0.6)
ax.set_axisbelow(True) # 讓格線藏在長條圖後方

# ==========================================
# 5. 在長條圖上方加入數值標籤 (值為 0 則不顯示)
# ==========================================
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 垂直往上偏 3 像素
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)

# ==========================================
# 6. 顯示並儲存圖表
# ==========================================
plt.tight_layout()

# 自動將圖片存到您的下載資料夾中
output_image_path = r"C:\Users\user\Downloads\RF_Top5_Wafer_Distribution.png"
plt.savefig(output_image_path, dpi=300)
print(f"✅ 圖表繪製完成！已高畫質儲存至：{output_image_path}")

plt.show()