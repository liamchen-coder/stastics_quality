# import os
# import pandas as pd
# import numpy as np
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# import warnings
# warnings.filterwarnings('ignore')

# # 1. 讀取最原始的降維特徵資料集
# data_path = r"C:\Users\user\Downloads\reduced_wafer_data.csv"
# df = pd.read_csv(data_path)

# print("="*90)
# print(" 啟動補充分析：K-Means 批次行為聚類與製程相似性(派工路徑)溯源")
# print("="*90)

# # =========================================================================
# # 步驟一：提取 10 個 LOT 的觀測值統計波動特徵
# # =========================================================================
# print("[Step 1] 正在提取各 LOT 之觀測值波動特徵 (均值、標準差、變異數、IQR)...")
# lot_stats = df.groupby('LOT')['OBSERVATION'].agg([
#     ('Mean', 'mean'),
#     ('Std', 'std'),
#     ('Variance', 'var'),
#     ('IQR', lambda x: np.percentile(x, 75) - np.percentile(x, 25))
# ]).reset_index()

# # 特徵標準化 (確保距離計算不受尺度影響)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(lot_stats.drop(columns=['LOT']))

# # =========================================================================
# # 步驟二：執行 K-Means 聚類 (分為 3 群)
# # =========================================================================
# print("[Step 2] 正在執行 K-Means 非監督式分群 (K=3)...")
# kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
# lot_stats['Cluster'] = kmeans.fit_predict(X_scaled)

# # =========================================================================
# # 步驟三：反向溯源 - 分析各群體為什麼相似？(找出他們共同愛用的機台)
# # =========================================================================
# print("[Step 3] 正在反向溯源各群體之「機台派工覆蓋率」...")

# # 抓出所有的機台欄位
# machine_cols = [col for col in df.columns if col.startswith('T')]

# # 將分群標籤合併回原始晶圓資料
# df_merged = df.merge(lot_stats[['LOT', 'Cluster']], on='LOT', how='left')

# # 計算每個 Cluster 裡，各機台的使用率 (該群有百分之多少的晶圓走過該機台)
# cluster_machine_usage = df_merged.groupby('Cluster')[machine_cols].mean()

# # 輸出每一群的詳細分析報告
# for cluster_id in range(3):
#     # 1. 該群包含哪些 LOT
#     cluster_lots = lot_stats[lot_stats['Cluster'] == cluster_id]['LOT'].tolist()
    
#     # 2. 該群的平均波動行為
#     c_mean = lot_stats[lot_stats['Cluster'] == cluster_id]['Mean'].mean()
#     c_var = lot_stats[lot_stats['Cluster'] == cluster_id]['Variance'].mean()
    
#     # 3. 找出該群最具代表性(使用率最高，且與其他群差異最大)的前 5 大機台
#     # 我們比較該群的使用率與「非該群」的平均使用率差異
#     usage_diff = cluster_machine_usage.loc[cluster_id] - cluster_machine_usage.drop(cluster_id).mean()
#     top_5_signature_machines = usage_diff.nlargest(5)
    
#     print(f"\n{'*'*30} 【 Cluster {cluster_id} 群體特徵報告 】 {'*'*30}")
#     print(f" 包含批次 (LOTs) : {', '.join(cluster_lots)}")
#     print(f" 群體波動行為    : 平均值 = {c_mean:.4f} | 變異數 = {c_var:.4f}")
#     print(f" 製程相似性溯源 (該群高度共用的特色機台):")
    
#     for mach, diff in top_5_signature_machines.items():
#         mach_usage = cluster_machine_usage.loc[cluster_id, mach] * 100
#         print(f"   -> 機台 {mach} : 該群體內使用率達 {mach_usage:.1f}% (顯著高於其他群體)")

# # =========================================================================
# # 步驟四：導出報表
# # =========================================================================
# output_path = r"C:\Users\user\Downloads\kmeans_lot_clustering_summary.csv"
# lot_stats.to_csv(output_path, index=False)

# print("\n" + "="*90)
# print(f" K-Means 批次聚類特徵報表已成功導出至：\n {output_path}")
# print("="*90)


# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import silhouette_score
# import warnings
# warnings.filterwarnings('ignore')

# # 1. 讀取原始資料
# data_path = r"C:\Users\user\Downloads\reduced_wafer_data.csv"
# df = pd.read_csv(data_path)

# # 2. 提取 10 個 LOT 的波動特徵
# lot_stats = df.groupby('LOT')['OBSERVATION'].agg([
#     ('Mean', 'mean'), ('Std', 'std'), ('Variance', 'var'),
#     ('IQR', lambda x: np.percentile(x, 75) - np.percentile(x, 25))
# ]).reset_index()

# # 3. 特徵標準化 (確保距離計算公平)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(lot_stats.drop(columns=['LOT']))

# # 4. 準備陣列來儲存測試結果 (因為只有 10 個 LOT，我們測試 K=2 到 K=6)
# k_range = range(2, 7)
# sse = []
# silhouette_scores = []

# # 5. 迴圈測試不同的 K 值
# for k in k_range:
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     labels = kmeans.fit_predict(X_scaled)
    
#     # 紀錄手肘法的 SSE (Inertia)
#     sse.append(kmeans.inertia_)
#     # 紀錄輪廓係數 (Silhouette Score)
#     silhouette_scores.append(silhouette_score(X_scaled, labels))

# # ==========================================
# #  6. 繪製證據圖表 (可直接貼入論文)
# # ==========================================
# fig, ax1 = plt.subplots(figsize=(10, 5))

# # 畫出手肘法 (SSE) 藍線
# color = 'tab:blue'
# ax1.set_xlabel('Number of clusters (K)', fontweight='bold')
# ax1.set_ylabel('Sum of Squared Errors (SSE)', color=color, fontweight='bold')
# ax1.plot(k_range, sse, marker='o', linestyle='-', color=color, linewidth=2)
# ax1.tick_params(axis='y', labelcolor=color)

# # 建立共用 X 軸的第二個 Y 軸，畫出輪廓係數 (紅線)
# ax2 = ax1.twinx()  
# color = 'tab:red'
# ax2.set_ylabel('Silhouette Score', color=color, fontweight='bold')
# ax2.plot(k_range, silhouette_scores, marker='s', linestyle='--', color=color, linewidth=2)
# ax2.tick_params(axis='y', labelcolor=color)

# # 自動找出輪廓係數最高的那一個 K
# best_k = k_range[np.argmax(silhouette_scores)]

# plt.title(f'K-Means Optimal K Selection\n(Best K based on Silhouette Score = {best_k})', fontsize=14, fontweight='bold')
# plt.grid(True, alpha=0.3)
# fig.tight_layout()

# # 另存圖檔到下載資料夾
# output_image_path = r"C:\Users\user\Downloads\K_Selection_Evidence.png"
# plt.savefig(output_image_path, dpi=300)
# print(f" 根據嚴謹的數學指標評估，建議將這 10 個 LOT 分成 【 {best_k} 】 群！")
# print(f" K 值選擇證據圖表已導出至：{output_image_path}")

# plt.show()
import os
import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# =========================================================================
#  核心演算法：Ohit2011 (正交貪婪演算法 OGA) & True Two-Hit
# =========================================================================
def Ohit2011(X, y, Kn=None, c1=5, HDIC_Type="HDHQ", const2=np.arange(0.5, 2.5, 0.5), const3=np.arange(0.2, 1.2, 0.2)):
    n, p = X.shape
    if Kn is None: Kn = max(1, min(math.floor(c1 * math.sqrt(n / math.log(p))), p))
    K = int(Kn)
    dy, dX = y - np.mean(y), X - np.mean(X, axis=0)
    Jhat, sigma2hat, XJhat = np.zeros(K, dtype=int), np.zeros(K), np.zeros((n, K))
    u = dy.copy()
    xnorms = np.sqrt(np.sum(dX**2, axis=0))
    xnorms[xnorms == 0] = 1e-8

    aSSE = np.abs(u.T @ dX) / xnorms
    Jhat[0] = np.argmax(aSSE)
    XJhat[:, 0] = dX[:, Jhat[0]] / np.sqrt(np.sum(dX[:, Jhat[0]]**2))
    u = u - XJhat[:, 0] * np.dot(XJhat[:, 0], u)
    sigma2hat[0] = np.mean(u**2)

    if K > 1:
        for k in range(1, K):
            aSSE = np.abs(u.T @ dX) / xnorms
            aSSE[Jhat[:k]] = -1  
            Jhat[k] = np.argmax(aSSE)
            proj = XJhat[:, :k] @ (XJhat[:, :k].T @ dX[:, Jhat[k]])
            rq = dX[:, Jhat[k]] - proj
            norm_rq = np.sqrt(np.sum(rq**2))
            XJhat[:, k] = rq / norm_rq if norm_rq > 1e-8 else rq
            u = u - XJhat[:, k] * np.dot(XJhat[:, k], u)
            sigma2hat[k] = np.mean(u**2)

    penalty = 1 if HDIC_Type == "HDAIC" else (math.log(n) if HDIC_Type == "HDBIC" else math.log(math.log(n)))
    kn_hat_now = -1
    result_all = []

    for c2 in const2:
        omega_n = c2 * penalty
        hdic = (n * np.log(sigma2hat + 1e-8)) + (np.arange(1, K + 1) * omega_n * math.log(p))
        kn_hat = np.argmin(hdic) + 1
        if kn_hat_now != kn_hat:
            for c3 in const3:
                benchmark = (n * math.log(sigma2hat[kn_hat-1] + 1e-8)) + (kn_hat * c3 * penalty * math.log(p))
                J_Trim = Jhat[:kn_hat].tolist()
                trim_pos = np.zeros(kn_hat, dtype=int)
                if kn_hat > 1:
                    for l in range(kn_hat):
                        JDrop1 = J_Trim[:l] + J_Trim[l+1:]
                        X_drop = dX[:, JDrop1]
                        beta, _, _, _ = np.linalg.lstsq(X_drop, dy, rcond=None)
                        HDICDrop1 = (n * math.log(np.mean((dy - X_drop @ beta)**2) + 1e-8)) + ((kn_hat - 1) * c3 * penalty * math.log(p))
                        if HDICDrop1 > benchmark: trim_pos[l] = 1
                    J_Trim_final = [J_Trim[i] for i in range(kn_hat) if trim_pos[i] == 1]
                else:
                    J_Trim_final = J_Trim.copy()
                if J_Trim_final not in result_all: result_all.append(J_Trim_final)
            kn_hat_now = kn_hat
    return {"Trim": result_all}

def True_Twohit(X, y, Kn=None, c1=5, HDIC_Type="HDHQ"):
    n, p = X.shape
    benchmark = float('inf')
    best_Trim_reg, best_Trim_dis, best_Beta, best_Alpha = [], [], [], []
    vs_reg = Ohit2011(X, y, Kn, c1, HDIC_Type)

    for variable_reg in vs_reg["Trim"]:
        nowX_reg = np.column_stack((np.ones(n), X[:, variable_reg])) if len(variable_reg) > 0 else np.ones((n, 1))
        beta_init, _, _, _ = np.linalg.lstsq(nowX_reg, y, rcond=None)
        u = y - nowX_reg @ beta_init
        u[np.abs(u) < 1e-4] = np.sign(u[np.abs(u) < 1e-4]) * 1e-4
        u[u == 0] = 1e-4
        newU = np.log(u**2)
        vs_dis = Ohit2011(X, newU, Kn, c1, HDIC_Type)

        for variable_dis in vs_dis["Trim"]:
            nowX_dis = np.column_stack((np.ones(n), X[:, variable_dis])) if len(variable_dis) > 0 else np.ones((n, 1))
            def L(theta):
                val = np.clip(nowX_dis @ theta[nowX_reg.shape[1]:], -50, 50) 
                return np.sum(val) + np.sum((y - nowX_reg @ theta[:nowX_reg.shape[1]])**2 / np.exp(val))

            alpha_init, _, _, _ = np.linalg.lstsq(nowX_dis, newU, rcond=None)
            try:
                paraEst = minimize(L, np.concatenate((beta_init, alpha_init)), method="L-BFGS-B")
                if paraEst.success or paraEst.status == 1:
                    df_model = len(variable_reg) + len(variable_dis)
                    penalty_term = 1 if HDIC_Type == "HDAIC" else (math.log(n) if HDIC_Type == "HDBIC" else math.log(math.log(n)))
                    hdicNow = L(paraEst.x) + (df_model * math.log(2*p) * penalty_term * 2)
                    if hdicNow < benchmark:
                        benchmark = hdicNow
                        best_Trim_reg, best_Trim_dis = variable_reg, variable_dis
                        best_Beta, best_Alpha = paraEst.x[:nowX_reg.shape[1]], paraEst.x[nowX_reg.shape[1]:]
            except Exception: continue
    return {"Trim_reg": best_Trim_reg, "Trim_dis": best_Trim_dis, "Beta": best_Beta, "Alpha": best_Alpha}

# =========================================================================
# 主程式：K-Means 自動分群 -> 轉換為 Super LOT 進行局部審判
# =========================================================================
data_path = r"C:\Users\user\Downloads\reduced_wafer_data.csv"
df = pd.read_csv(data_path)

print("="*90)
print(" 啟動分析：K-Means 批次聚類與 Super LOT (群體樣本池化) 溯源審判")
print("="*90)

# --- 步驟一：K-Means 分群 (K=3) ---
lot_stats = df.groupby('LOT')['OBSERVATION'].agg(['mean', 'std', 'var', lambda x: np.percentile(x, 75) - np.percentile(x, 25)]).reset_index()
lot_stats.columns = ['LOT', 'Mean', 'Std', 'Variance', 'IQR']
X_scaled = StandardScaler().fit_transform(lot_stats.drop(columns=['LOT']))
lot_stats['Cluster'] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X_scaled)

cluster_mapping = {f"Cluster {c}": lot_stats[lot_stats['Cluster'] == c]['LOT'].tolist() for c in range(3)}

# --- 步驟二：Super LOT 審判 ---
machine_cols = [col for col in df.columns if col.startswith('T')]
hdic_methods = ["HDAIC", "HDHQ", "HDBIC"] #  這裡已經為您補上 HDHQ

for cluster_name, lots in cluster_mapping.items():
    print(f"\n\n{'*'*25} 【 Super LOT 審判目標：{cluster_name} 】 {'*'*25}")
    print(f"  包含批次: {lots} (將合併視為單一巨大樣本池)")
    
    df_cluster = df[df['LOT'].isin(lots)].copy()
    X, y = df_cluster[machine_cols].to_numpy(), df_cluster['OBSERVATION'].to_numpy()
    
    # 新增機制：抓出覆蓋率 > 95% 的「群體基底機台」
    coverage = X.mean(axis=0)
    signature_idx = np.where(coverage > 0.95)[0]
    if len(signature_idx) > 0:
        print("\n  [第一階段：抓出該 Super LOT 絕對共用的基底機台]")
        for idx in signature_idx:
            print(f"      - 機台: {machine_cols[idx]} | 覆蓋率: {coverage[idx]*100:.1f}% (該群體共同經歷的潛在污染源)")

    # 剃除變異為 0 的常數機台，準備進入 Two-Hit
    valid_cols_idx = np.where((coverage > 0) & (coverage < 1))[0]
    if len(valid_cols_idx) == 0: continue
    
    X_valid = X[:, valid_cols_idx]
    machine_names = np.array(machine_cols)[valid_cols_idx]

    print("\n  [ 第二階段：執行 Two-Hit 揪出群內作亂戰犯]")
    for method in hdic_methods:
        print(f"   -> 採用準則：{method}")
        res = True_Twohit(X_valid, y, HDIC_Type=method)
        
        # Location Model
        reg_str = ", ".join([f"{machine_names[i]} (Beta: {c:.4f})" for i, c in zip(res["Trim_reg"], res["Beta"][1:])])
        print(f"      均值模型 (均值漂移): {reg_str if reg_str else '(無)'}")
        
        # Dispersion Model
        dis_str = ", ".join([f"{machine_names[i]} (Alpha: {c:.4f})" for i, c in zip(res["Trim_dis"], res["Alpha"][1:])])
        print(f"      變異模型 (異質波動): {dis_str if dis_str else '(無)'}")