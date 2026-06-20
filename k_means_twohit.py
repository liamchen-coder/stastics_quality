# import os
# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LassoCV
# import warnings
# warnings.filterwarnings('ignore')

# data_path = r"C:\Users\user\Downloads\reduced_wafer_data.csv"
# df = pd.read_csv(data_path)

# print("="*90)
# print(" 啟動高階戰術：基於 K-Means 分群之局部 Two-Hit 演算法 (Local Location-Dispersion Modeling)")
# print("="*90)

# # 2. 根據你的 K-Means 結果，手動映射各 Cluster 包含的 LOT
# cluster_mapping = {
#     "Cluster 0 (健康基準群)": ['LOT1', 'LOT2', 'LOT3', 'LOT7'],
#     "Cluster 1 (中度偏移群)": ['LOT4', 'LOT5', 'LOT6', 'LOT8'],
#     "Cluster 2 (嚴重暴走群)": ['LOT9', 'LOT10']
# }

# # 抓出所有的機台欄位 (自變數 X)
# machine_cols = [col for col in df.columns if col.startswith('T')]

# # 3. 迴圈對每一個 Cluster 進行獨立的 Two-Hit 審判
# for cluster_name, lots in cluster_mapping.items():
#     print(f"\n{'*'*30} 【 審判目標：{cluster_name} 】 {'*'*30}")
#     print(f" 包含批次: {lots}")
    
#     # 篩選出屬於該 Cluster 的晶圓資料
#     df_cluster = df[df['LOT'].isin(lots)].copy()
    
#     X = df_cluster[machine_cols].to_numpy()
#     y = df_cluster['OBSERVATION'].to_numpy()
    
#     # 確保該群體內有足夠的樣本進行運算 (避免單一機台全為0或1導致無變異)
#     # 過濾掉在該群體內使用率為 0% 或 100% 的機台 (因為這類機台無法提供變異解釋力)
#     valid_cols_idx = np.where((X.mean(axis=0) > 0) & (X.mean(axis=0) < 1))[0]
#     X_valid = X[:, valid_cols_idx]
#     valid_machine_names = np.array(machine_cols)[valid_cols_idx]
    
#     if len(valid_cols_idx) == 0:
#         print("  ⚠️ 該群體內機台特徵無變異(皆為 0 或皆為 1)，無法執行 Two-Hit。")
#         continue

#     # ==========================================================
#     # Hit 1: 均值模型 (Location Model) - 尋找導致觀測值平均數偏移的機台
#     # ==========================================================
#     # 使用 LassoCV 自動尋找最佳懲罰力度 (5折交叉驗證)
#     lasso_mean = LassoCV(cv=5, random_state=42, max_iter=10000)
#     lasso_mean.fit(X_valid, y)
    
#     # 計算預測值與殘差
#     y_pred = lasso_mean.predict(X_valid)
#     residuals = y - y_pred
    
#     # 抓出均值模型中係數不為 0 的機台
#     hit1_machines = []
#     for i, coef in enumerate(lasso_mean.coef_):
#         if coef != 0:
#             hit1_machines.append((valid_machine_names[i], coef))
            
#     # ==========================================================
#     # Hit 2: 變異數模型 (Dispersion Model) - 尋找導致變異數(殘差)放大的機台
#     # ==========================================================
#     # 計算對數平方殘差 log(e^2 + epsilon) 作為變異數目標值
#     # 加上微小值 1e-6 避免 log(0) 錯誤
#     log_sq_residuals = np.log(residuals**2 + 1e-6)
    
#     lasso_var = LassoCV(cv=5, random_state=42, max_iter=10000)
#     lasso_var.fit(X_valid, log_sq_residuals)
    
#     # 抓出變異數模型中係數不為 0 的機台
#     hit2_machines = []
#     for i, coef in enumerate(lasso_var.coef_):
#         if coef != 0:
#             hit2_machines.append((valid_machine_names[i], coef))

#     # ==========================================================
#     # 印出該 Cluster 的 Two-Hit 審判結果
#     # ==========================================================
#     print("\n   [Hit 1 均值模型] 影響該群基準線偏移的核心機台：")
#     if not hit1_machines:
#         print("     (無顯著機台，群內基準線穩定)")
#     else:
#         for mach, coef in sorted(hit1_machines, key=lambda x: abs(x[1]), reverse=True)[:5]:
#             direction = "正向偏移" if coef > 0 else "負向偏移"
#             print(f"     - 機台 {mach} : 係數 {coef:.4f} ({direction})")

#     print("\n   [Hit 2 變異數模型] 放大該群體製程波動(Variance)的核心機台：")
#     if not hit2_machines:
#         print("     (無顯著機台，群內無異常波動源)")
#     else:
#         for mach, coef in sorted(hit2_machines, key=lambda x: abs(x[1]), reverse=True)[:5]:
#             # 變異數模型係數為正，代表經過該機台會放大波動
#             direction = "放大波動 (極危險)" if coef > 0 else "縮小波動 (穩定)"
#             print(f"     - 機台 {mach} : 係數 {coef:.4f} ({direction})")
#     print("-" * 80)

# print("\n 分群局部 Two-Hit 分析完畢！")

import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# =========================================================================
# Core Algorithm 1: Ohit2011 (Orthogonal Greedy Algorithm + HDIC Trimming)
# =========================================================================
def Ohit2011(X, y, Kn=None, c1=5, HDIC_Type="HDHQ", const2=np.arange(0.5, 2.5, 0.5), const3=np.arange(0.2, 1.2, 0.2)):
    n, p = X.shape
    if Kn is None:
        Kn = max(1, min(math.floor(c1 * math.sqrt(n / math.log(p))), p))
    K = int(Kn)

    dy = y - np.mean(y)
    dX = X - np.mean(X, axis=0)
    Jhat = np.zeros(K, dtype=int)
    sigma2hat = np.zeros(K)
    XJhat = np.zeros((n, K))
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

    if HDIC_Type == "HDAIC": penalty = 1
    elif HDIC_Type == "HDBIC": penalty = math.log(n)
    else: penalty = math.log(math.log(n))

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
                        uDrop1 = dy - X_drop @ beta
                        HDICDrop1 = (n * math.log(np.mean(uDrop1**2) + 1e-8)) + ((kn_hat - 1) * c3 * penalty * math.log(p))
                        if HDICDrop1 > benchmark:
                            trim_pos[l] = 1
                    J_Trim_final = [J_Trim[i] for i in range(kn_hat) if trim_pos[i] == 1]
                else:
                    J_Trim_final = J_Trim.copy()

                if J_Trim_final not in result_all:
                    result_all.append(J_Trim_final)
            kn_hat_now = kn_hat

    return {"OGA": Jhat, "Trim": result_all}


# =========================================================================
# Core Algorithm 2: Joint Likelihood Optimization (from Twohit.r)
# =========================================================================
def True_Twohit(X, y, Kn=None, c1=5, HDIC_Type="HDHQ"):
    n, p = X.shape
    benchmark = float('inf')
    best_Trim_reg, best_Trim_dis = [], []
    best_Beta, best_Alpha = [], []

    vs_reg = Ohit2011(X, y, Kn, c1, HDIC_Type)
    variable_all_reg = vs_reg["Trim"]

    for variable_reg in variable_all_reg:
        nowX_reg = np.column_stack((np.ones(n), X[:, variable_reg])) if len(variable_reg) > 0 else np.ones((n, 1))
        beta_init, _, _, _ = np.linalg.lstsq(nowX_reg, y, rcond=None)
        u = y - nowX_reg @ beta_init
        
        pos = np.abs(u) < 1e-4
        u[pos] = np.sign(u[pos]) * 1e-4
        u[u == 0] = 1e-4
        newU = np.log(u**2)

        vs_dis = Ohit2011(X, newU, Kn, c1, HDIC_Type)
        variable_all_dis = vs_dis["Trim"]

        for variable_dis in variable_all_dis:
            nowX_dis = np.column_stack((np.ones(n), X[:, variable_dis])) if len(variable_dis) > 0 else np.ones((n, 1))

            def L(theta):
                len_beta = nowX_reg.shape[1]
                theta_beta = theta[:len_beta]
                theta_alpha = theta[len_beta:]
                # Prevent overflow in exp
                val = np.clip(nowX_dis @ theta_alpha, -50, 50) 
                return np.sum(val) + np.sum((y - nowX_reg @ theta_beta)**2 / np.exp(val))

            alpha_init, _, _, _ = np.linalg.lstsq(nowX_dis, newU, rcond=None)
            initial_estimate = np.concatenate((beta_init, alpha_init))

            try:
                paraEst = minimize(L, initial_estimate, method="L-BFGS-B")
                if paraEst.success or paraEst.status == 1:
                    ll_val = L(paraEst.x)
                    df = len(variable_reg) + len(variable_dis)
                    
                    if HDIC_Type == "HDAIC": penalty_term = 1
                    elif HDIC_Type == "HDBIC": penalty_term = math.log(n)
                    else: penalty_term = math.log(math.log(n))

                    hdicNow = ll_val + (df * math.log(2*p) * penalty_term * 2)

                    if hdicNow < benchmark:
                        benchmark = hdicNow
                        best_Trim_reg = variable_reg
                        best_Trim_dis = variable_dis
                        best_Beta = paraEst.x[:nowX_reg.shape[1]]
                        best_Alpha = paraEst.x[nowX_reg.shape[1]:]
            except Exception:
                continue

    return {"Trim_reg": best_Trim_reg, "Trim_dis": best_Trim_dis, "Beta": best_Beta, "Alpha": best_Alpha}


# =========================================================================
# Main Execution: K-Means Clustering + Two-Hit Analysis
# =========================================================================
data_path = r"C:\Users\user\Downloads\reduced_wafer_data.csv"
df = pd.read_csv(data_path)

cluster_mapping = {
    "Cluster 0": ['LOT1', 'LOT2', 'LOT3', 'LOT7'],
    "Cluster 1": ['LOT4', 'LOT5', 'LOT6', 'LOT8'],
    "Cluster 2": ['LOT9', 'LOT10']
}

machine_cols = [col for col in df.columns if col.startswith('T')]
hdic_methods = ["HDAIC", "HDHQ", "HDBIC"]

print("="*90)
print("Execution Started: K-Means Clustering + True Two-Hit with HDIC Evaluation")
print("="*90)

for cluster_name, lots in cluster_mapping.items():
    print(f"\n\n{'*'*20} Target: {cluster_name} {'*'*20}")
    print(f"Included LOTs: {lots}")
    
    df_cluster = df[df['LOT'].isin(lots)].copy()
    X = df_cluster[machine_cols].to_numpy()
    y = df_cluster['OBSERVATION'].to_numpy()
    
    valid_cols_idx = np.where((X.mean(axis=0) > 0) & (X.mean(axis=0) < 1))[0]
    if len(valid_cols_idx) == 0:
        print("Warning: No variance in machine features for this cluster. Skipping.")
        continue
        
    X_valid = X[:, valid_cols_idx]
    machine_names = np.array(machine_cols)[valid_cols_idx]

    for method in hdic_methods:
        print(f"\nExecuting Two-Hit Algorithm -> Criterion: {method}")
        res = True_Twohit(X_valid, y, HDIC_Type=method)
        
        print(f"  [1. Location Model] Identified {len(res['Trim_reg'])} critical machines:")
        if not res["Trim_reg"]:
            print("      (None)")
        else:
            for idx_in_valid, coef in zip(res["Trim_reg"], res["Beta"][1:]):
                mach = machine_names[idx_in_valid]
                print(f"      - Machine: {mach} | Beta Coefficient: {coef:.6f}")

        print(f"  [2. Dispersion Model] Identified {len(res['Trim_dis'])} critical machines:")
        if not res["Trim_dis"]:
            print("      (None)")
        else:
            for idx_in_valid, coef in zip(res["Trim_dis"], res["Alpha"][1:]):
                mach = machine_names[idx_in_valid]
                print(f"      - Machine: {mach} | Alpha Coefficient: {coef:.6f}")
                
    print("-" * 80)