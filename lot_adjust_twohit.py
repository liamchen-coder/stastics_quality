# import numpy as np
# import pandas as pd
# from scipy.optimize import minimize

# def Ohit2011(X, y, Kn=None, c1=5, HDIC_Type="HDHQ", const2=None, const3=None):
#     """
#     正交格點追蹤法 (OGA) 搭配高維度資訊準則 (HDIC) 與剪裁 (Trim) 機制
#     """
#     if const2 is None:
#         const2 = np.arange(0.5, 2.1, 0.5)
#     if const3 is None:
#         const3 = np.arange(0.2, 1.1, 0.2)
        
#     n, p = X.shape
#     K = int(max(1, min(np.floor(c1 * np.sqrt(n / np.log(p))), p))) if Kn is None else int(Kn)
    
#     dy = y - np.mean(y)
#     # 欄位中心化
#     dX = X - np.mean(X, axis=0)
    
#     Jhat = np.zeros(K, dtype=int)
#     sigma2hat = np.zeros(K)
#     XJhat = np.zeros((n, K))
#     u = dy.copy().reshape(-1, 1)
    
#     xnorms = np.sqrt(np.sum(dX**2, axis=0))
#     xnorms[xnorms == 0] = 1e-12
    
#     aSSE = np.abs(u.T @ dX).flatten() / xnorms
#     Jhat[0] = np.argmax(aSSE)
#     XJhat[:, 0] = dX[:, Jhat[0]] / np.sqrt(np.sum(dX[:, Jhat[0]]**2))
#     u = u - (XJhat[:, 0:1] @ XJhat[:, 0:1].T @ u)
#     sigma2hat[0] = np.mean(u**2)
    
#     if K > 1:
#         for k in range(1, K):
#             aSSE = np.abs(u.T @ dX).flatten() / xnorms
#             aSSE[Jhat[0:k]] = 0  
#             Jhat[k] = np.argmax(aSSE)
            
#             qk = dX[:, Jhat[k]:Jhat[k]+1]
#             X_selected = XJhat[:, 0:k]
#             rq = qk - (X_selected @ X_selected.T @ qk)
            
#             rq_norm = np.sqrt(np.sum(rq**2))
#             if rq_norm == 0:
#                 rq_norm = 1e-12
                
#             XJhat[:, k] = (rq / rq_norm).flatten()
#             u = u - (XJhat[:, k:k+1] @ XJhat[:, k:k+1].T @ u)
#             sigma2hat[k] = np.mean(u**2)
            
#     # 設定 HDIC 懲罰項
#     if HDIC_Type == "HDAIC":
#         penalty = 1.0
#     elif HDIC_Type == "HDBIC":
#         penalty = np.log(n)
#     elif HDIC_Type == "HDHQ":
#         penalty = np.log(np.log(n))
#     else:
#         penalty = np.log(np.log(n))
        
#     kn_hat_now = -1
#     result_all = []
    
#     for c2 in const2:
#         omega_n = c2 * penalty
#         hdic = (n * np.log(sigma2hat)) + (np.arange(1, K + 1) * omega_n * np.log(p))
#         kn_hat = np.argmin(hdic)
        
#         if kn_hat_now != kn_hat:
#             for c3 in const3:
#                 benchmark_trim = (n * np.log(sigma2hat[kn_hat])) + ((kn_hat + 1) * c3 * penalty * np.log(p))
#                 J_Trim = list(Jhat[0:kn_hat+1])
#                 trim_pos = np.zeros(len(J_Trim))
                
#                 if len(J_Trim) > 1:
#                     for l in range(len(J_Trim)):
#                         JDrop1 = [J_Trim[m] for m in range(len(J_Trim)) if m != l]
#                         dX_drop = dX[:, JDrop1]
#                         coef_drop, _, _, _ = np.linalg.lstsq(dX_drop, dy, rcond=None)
#                         uDrop1 = dy - dX_drop @ coef_drop
                        
#                         HDICDrop1 = (n * np.log(np.mean(uDrop1**2))) + (len(JDrop1) * c3 * penalty * np.log(p))
#                         if HDICDrop1 > benchmark_trim:
#                             trim_pos[l] = 1
                            
#                     J_Trim = [J_Trim[idx] for idx in range(len(J_Trim)) if trim_pos[idx] == 1]
                
#                 res = list(sorted(J_Trim)) + [0] * (K - len(J_Trim))
#                 result_all.append(res)
#             kn_hat_now = kn_hat
            
#     return {"OGA": Jhat, "Trim": np.array(result_all)}

# def Twohit(X, y, Kn=None, c1=5, HDIC_Type="HDHQ", const2=None, const3=None):
#     """
#     Chiou, Guo, and Ing (2020) 雙重打擊故障診斷核心演算法
#     """
#     if const2 is None:
#         const2 = np.arange(0.5, 2.1, 0.5)
#     if const3 is None:
#         const3 = np.arange(0.2, 1.1, 0.2)
        
#     n, p = X.shape
#     benchmark = np.inf
    
#     vs_reg = Ohit2011(X, y, Kn, c1, HDIC_Type, const2, const3)
#     variable_all_reg = vs_reg["Trim"]
#     num_all_reg = variable_all_reg.shape[0]
    
#     Trim_reg, Trim_dis, Beta, Alpha = [], [], None, None
    
#     for i in range(num_all_reg):
#         row_reg = variable_all_reg[i, :]
#         variable_reg = row_reg[row_reg != 0]
        
#         if len(variable_reg) > 0:
#             nowX_reg = np.column_stack([np.ones(n), X[:, variable_reg]])
#         else:
#             nowX_reg = np.ones((n, 1))
            
#         coef_reg, _, _, _ = np.linalg.lstsq(nowX_reg, y, rcond=None)
#         u = y - nowX_reg @ coef_reg
        
#         u[np.abs(u)**2 < 10**(-8)] = np.sign(u[np.abs(u)**2 < 10**(-8)]) * (10**(-4))
#         newU = np.log(u**2)
        
#         vs_dis = Ohit2011(X, newU, Kn, c1, HDIC_Type, const2, const3)
#         variable_all_dis = vs_dis["Trim"]
#         num_all_dis = variable_all_dis.shape[0]
        
#         for j in range(num_all_dis):
#             row_dis = variable_all_dis[j, :]
#             variable_dis = row_dis[row_dis != 0]
            
#             if len(variable_dis) > 0:
#                 nowX_dis = np.column_stack([np.ones(n), X[:, variable_dis]])
#             else:
#                 nowX_dis = np.ones((n, 1))
                
#             len_reg = len(variable_reg)
#             len_dis = len(variable_dis)
            
#             if len_reg == 0 and len_dis == 0:
#                 def L(theta):
#                     return np.sum(theta[1]) + np.sum((y - theta[0])**2 / np.exp(theta[1]))
#                 initial_estimate = [np.mean(y), np.mean(newU)]
                
#             elif len_reg > 0 and len_dis == 0:
#                 def L(theta):
#                     beta_p = theta[0 : len_reg + 1]
#                     alpha_p = theta[len_reg + 1]
#                     return np.sum(alpha_p) + np.sum((y - nowX_reg @ beta_p)**2 / np.exp(alpha_p))
#                 coef_init, _, _, _ = np.linalg.lstsq(nowX_reg, y, rcond=None)
#                 initial_estimate = list(coef_init) + [np.mean(newU)]
                
#             elif len_reg == 0 and len_dis > 0:
#                 def L(theta):
#                     beta_p = theta[0]
#                     alpha_p = theta[1:]
#                     return np.sum(nowX_dis @ alpha_p) + np.sum((y - beta_p)**2 / np.exp(nowX_dis @ alpha_p))
#                 coef_dis_init, _, _, _ = np.linalg.lstsq(nowX_dis, newU, rcond=None)
#                 initial_estimate = [np.mean(y)] + list(coef_dis_init)
                
#             else:
#                 def L(theta):
#                     beta_p = theta[0 : len_reg + 1]
#                     alpha_p = theta[len_reg + 1 :]
#                     return np.sum(nowX_dis @ alpha_p) + np.sum((y - nowX_reg @ beta_p)**2 / np.exp(nowX_dis @ alpha_p))
#                 coef_init, _, _, _ = np.linalg.lstsq(nowX_reg, y, rcond=None)
#                 coef_dis_init, _, _, _ = np.linalg.lstsq(nowX_dis, newU, rcond=None)
#                 initial_estimate = list(coef_init) + list(coef_dis_init)
            
#             try:
#                 paraEst = minimize(L, initial_estimate, method="L-BFGS-B")
#                 if not paraEst.success:
#                     continue
                
#                 hdicNow = paraEst.fun + ((len_reg + len_dis) * np.log(2 * p) * np.log(np.log(n)) * 2)
                
#                 if hdicNow < benchmark:
#                     Trim_reg = list(variable_reg)
#                     Trim_dis = list(variable_dis)
#                     Beta = paraEst.x[0 : len_reg + 1]
#                     Alpha = paraEst.x[len_reg + 1 :]
#                     benchmark = hdicNow
#             except:
#                 continue

#     return {
#         "Trim_reg": Trim_reg, 
#         "Trim_dis": Trim_dis, 
#         "Beta": Beta, 
#         "Alpha": Alpha
#     }

# # ==============================================================================
# # 測試執行框架（自動循環跑完 HDAIC, HDBIC, HDHQ 三種模型）
# # ==============================================================================
# if __name__ == "__main__":
#     # 請替換成你實際的 CSV 檔案路徑
#     data_path = r"C:\Users\user\Downloads\reduced_wafer_data_with_fitted_and_residual.csv"
    
#     try:
#         df = pd.read_csv(data_path)
#         print(f"成功讀取資料集，資料形狀為: {df.shape}")
        
#         # 提取應變數：Lot-adjusted Residual 殘差
#         y_data = df['LOT_ADJUSTED_RESIDUAL'].to_numpy()
        
#         # 提取自變數：所有以 'T' 開頭的機台特徵
#         X_df = df[[col for col in df.columns if col.startswith('T')]]
#         X_data = X_df.to_numpy()
#         t_variables_names = list(X_df.columns)
        
#         # 定義要測試的三種不同嚴格程度的資訊準則
#         criteria_list = ["HDAIC", "HDBIC", "HDHQ"]
        
#         print("\n=== 開始進行三種高維度資訊準則 (HDIC) 的變數選取對比 ===")
        
#         for criterion in criteria_list:
#             print(f"\n🚀 正在執行大師級 Two-Hit 演算法 -> 【 準則類型: {criterion} 】...")
            
#             # 呼叫 Twohit 函數並傳入當前的準則
#             results = Twohit(X_data, y_data, HDIC_Type=criterion)
            
#             print("-"*60)
#             print(f"📊 【準則 {criterion} 執行成果】")
#             print("-"*60)
            
#             # 1. 印出該準則下的均值模型結果
#             print(f" 1. 均值模型 (Regression Model) 共選出 {len(results['Trim_reg'])} 個關鍵機台：")
#             if len(results['Trim_reg']) == 0:
#                 print("    -> 無顯著影響均值之機台特徵。")
#             else:
#                 for idx, var_idx in enumerate(results['Trim_reg']):
#                     print(f"    機台: {t_variables_names[var_idx]} | 迴歸係數 Beta: {results['Beta'][idx+1]:.6f}")
#             if results['Beta'] is not None:
#                 print(f"    (截距項 Beta0: {results['Beta'][0]:.6f})")
                
#             print()
            
#             # 2. 印出該準則下的分散模型結果
#             print(f" 2. 變異數模型 (Dispersion Model) 共選出 {len(results['Trim_dis'])} 個關鍵機台：")
#             if len(results['Trim_dis']) == 0:
#                 print("    -> 無顯著放大製程變異之機台特徵。")
#             else:
#                 for idx, var_idx in enumerate(results['Trim_dis']):
#                     print(f"    機台: {t_variables_names[var_idx]} | 異質波動係數 Alpha: {results['Alpha'][idx+1]:.6f}")
#             if results['Alpha'] is not None:
#                 print(f"    (截距項 Alpha0: {results['Alpha'][0]:.6f})")
                
#             print("="*60)
            
#         print("\n🎉 所有準則測試完畢！你可以直接對比上方印出的結果進行報告撰寫。")
        
#     except FileNotFoundError:
#         print(f"錯誤：找不到路徑下的資料檔案，請確認 {data_path} 是否正確。")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 讀取資料與設定特定機台
# ==========================================
# ⚠️ 請替換成你電腦中實際的 CSV 檔案路徑
data_path = r"C:\Users\user\Downloads\reduced_wafer_data.csv"

try:
    df = pd.read_csv(data_path)
    print(f"✅ 成功讀取資料集，資料形狀為: {df.shape}")
    
    # 設定你要觀察的 3 台特定機台
    target_machines = ['T0447', 'T0872', 'T1202']
    
    # 檢查這些機台是否都在資料集中
    missing_machines = [m for m in target_machines if m not in df.columns]
    if missing_machines:
        print(f"⚠️ 警告：資料集中找不到以下機台欄位：{missing_machines}")
    else:
        # ==========================================
        # 2. 資料分組與排序處理
        # ==========================================
        # 計算各 LOT 經過這三台機台的總片數
        lot_counts = df.groupby('LOT')[target_machines].sum()
        
        # 自訂排序規則：確保 X 軸是完美的 LOT1 ~ LOT10 順序
        def extract_lot_num(lot_str):
            try:
                return int(str(lot_str).upper().replace('LOT', ''))
            except:
                return 0
                
        sorted_idx = lot_counts.index.map(extract_lot_num).argsort()
        lot_counts = lot_counts.iloc[sorted_idx]
        
        # ==========================================
        # 3. 圖表設定與繪製
        # ==========================================
        # 設定中文字體 (Windows 預設微軟正黑體)
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ax = plt.subplots(figsize=(14, 6))
        x = np.arange(len(lot_counts.index))
        
        # 3 台機台的寬度與並排偏移量設定
        width = 0.25 
        offsets = [-width, 0, width]
        
        # 設定 3 種具備專業質感的對比色
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 
        
        # 繪製長條圖
        for i, machine in enumerate(target_machines):
            bars = ax.bar(x + offsets[i], lot_counts[machine], width, label=machine, 
                          color=colors[i], edgecolor='black', linewidth=0.8)
            
            # 加上數值標籤 (值大於 0 才顯示)
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(f'{int(height)}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3), textcoords="offset points",
                                ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # ==========================================
        # 4. 標籤、標題與格線美化
        # ==========================================
        ax.set_ylabel('行經晶圓片數 (片)', fontsize=12, fontweight='bold')
        ax.set_xlabel('生產批次名稱 (LOT)', fontsize=12, fontweight='bold')
        ax.set_title('各生產批次 (LOT 1~10 順序) 行經指定機台之晶圓數量分佈圖', fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(lot_counts.index, fontsize=12)
        
        # 自動調整 Y 軸上限，留出空間給數值標籤
        max_val = lot_counts.max().max()
        ax.set_ylim(0, max_val * 1.2)
        
        ax.legend(fontsize=12, loc='upper right')
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        
        # ==========================================
        # 5. 存檔與顯示
        # ==========================================
        output_image_path = r"C:\Users\user\Downloads\Specific_3_Machines_Distribution.png"
        plt.savefig(output_image_path, dpi=300)
        print(f"✅ 高畫質圖表已成功繪製並儲存至：{output_image_path}")
        
        plt.show()

except FileNotFoundError:
    print(f"❌ 錯誤：找不到路徑下的資料檔案，請確認 {data_path} 是否正確。")