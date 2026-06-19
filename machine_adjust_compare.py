import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import pingouin as pg
import warnings
warnings.filterwarnings('ignore')

# 1. 定義三個去噪完畢的檔案路徑與其對應的模型名稱
file_dict = {
    "Ridge": r"C:\Users\user\Downloads\reduced_wafer_data_ridge_adjusted.csv",
    "PCA": r"C:\Users\user\Downloads\reduced_wafer_data_pca_adjusted.csv",
    "SVR": r"C:\Users\user\Downloads\reduced_wafer_data_svr_adjusted.csv"
}

print("="*90)
print(" 啟動後半段整合審判：4.2.5 Isolation Forest 異常排行 與 4.2.6 Welch's ANOVA + Games-Howell")
print("="*90)

# 開始對 Ridge, PCA, SVR 的殘差結果進行迴圈並行分析
for model_name, file_path in file_dict.items():
    if not os.path.exists(file_path):
        print(f"❌ 找不到 {model_name} 的檔案，請確認路徑：{file_path}")
        continue
        
    print(f"\n{'*'*25} 【 軸線方案：{model_name} Machine-adjusted Residual 】 {'*'*25}\n")
    
    # 讀取數據
    df = pd.read_csv(file_path)
    
    # =========================================================================
    # 🤖 4.2.5 Isolation Forest (孤立森林異常檢測 - 尋找影響最多之 LOT)
    # =========================================================================
    print(f" [4.2.5 Isolation Forest] 正在利用二元樹隨機切分空間，計算晶圓個體之異常得分...")
    
    # 提取完全去除機台干擾後的淨殘差作為輸入特徵 (轉為 numpy 二維陣列)
    X_residual = df[['MACHINE_ADJUSTED_RESIDUAL']].to_numpy()
    
    # 宣告孤立森林模型 (contamination='auto' 自動調節，並固定隨機種子確保結果可重現)
    if_model = IsolationForest(contamination='auto', random_state=42)
    
    # 修正處：讓模型先進行數據學習與空間切分 (Fit)
    if_model.fit(X_residual)
    
    # 計算異常分數 (Anomaly Score)，取負號使其變為「正分數」，分數越高越異常
    df['ANOMALY_SCORE'] = -if_model.score_samples(X_residual)
    
    # 計算每個 LOT 內所有晶圓的平均異常分數，並由大到小排序，建立「受影響最多之 LOT 排行榜」
    lot_anomaly_ranking = df.groupby('LOT')['ANOMALY_SCORE'].mean().reset_index()
    lot_anomaly_ranking = lot_anomaly_ranking.sort_values(by='ANOMALY_SCORE', ascending=False).reset_index(drop=True)
    
    print("\n 【孤立森林：受環境/原料影響最多之 LOT 異常嚴重度排行榜】:")
    print(lot_anomaly_ranking.to_string(index=False))
    print(f"  結論: 在 {model_name} 殘差空間下，受非機台系統性因素干擾最嚴重的頭號戰犯為 【 {lot_anomaly_ranking['LOT'].iloc[0]} 】")
    print("-"*80)
    
    # =========================================================================
    # 🎯 4.2.6 Welch's ANOVA + Games-Howell (統計顯著性審判)
    # =========================================================================
    print(f" [4.2.6 假設檢定] 正在執行 Welch's ANOVA (對抗異質變異數與小樣本)...")
    
    # 執行 Welch's ANOVA
    welch_res = pg.welch_anova(data=df, dv='MACHINE_ADJUSTED_RESIDUAL', between='LOT')
    p_value = welch_res['p-unc'].values[0]
    f_stat = welch_res['F'].values[0]
    df_num = welch_res['ddof1'].values[0]
    df_den = welch_res['ddof2'].values[0]
    
    print(f"\n Welch's ANOVA 檢定報告:")
    print(f"  -> F-Statistic ({df_num}, {df_den:.2f}) = {f_stat:.4f}")
    print(f"  -> P-value = {p_value:.6e}")
    
    if p_value < 0.05:
        print("  ->  結論: 顯著！在完全排除機台污染後，10 個 LOT 之間仍存在極為顯著的基線波動差異。")
        print(f"  ->  正在啟動 Games-Howell 45 次成對多重比較審判...")
        
        # 執行 Games-Howell 事後檢定
        games_howell_res = pg.pairwise_gameshowell(data=df, dv='MACHINE_ADJUSTED_RESIDUAL', between='LOT')
        
        # 篩選出真正達到統計顯著（pval < 0.05）的異常配對組合
        sig_pairs = games_howell_res[games_howell_res['pval'] < 0.05]
        
        print(f"\n Games-Howell 顯著異常對比名單 (共抓出 {len(sig_pairs)} 組顯著差異配對):")
        if not sig_pairs.empty:
            print(sig_pairs[['A', 'B', 'diff', 'se', 'T', 'pval', 'hedges']].to_string(index=False))
            
            # 自動計算哪一個 LOT 出現次數最多，與大家割裂最嚴重
            all_sig_lots = pd.concat([sig_pairs['A'], sig_pairs['B']])
            lot_counts = all_sig_lots.value_counts()
            print(f"\n 【統計檢定：批次割裂/異常排名】:")
            for lot_id, count in lot_counts.items():
                print(f"  -> {lot_id} 與其他批次產生顯著基線裂解 {count} 次")
        else:
            print("  ->  雖然整體顯著，但兩兩比較未達 Games-Howell 保守臨界值門檻。")
            
        # =========================================================================
        # 💾 輸出完整分析成果至 Downloads 資料夾
        # =========================================================================
        output_path_if = f"C:\\Users\\user\\Downloads\\{model_name}_isolation_forest_ranking.csv"
        lot_anomaly_ranking.to_csv(output_path_if, index=False)
        
        output_path_gh = f"C:\\Users\\user\\Downloads\\{model_name}_games_howell_results.csv"
        games_howell_res.to_csv(output_path_gh, index=False)
        
        print(f"\n數據報告已另外獨立導出至您的 Downloads 資料夾：")
        print(f"  1. Isolation Forest 異常分數排行榜 -> {output_path_if}")
        print(f"  2. Games-Howell 完整 45 組成對矩陣  -> {output_path_gh}")
        
    else:
        print("  ->  結論: 不顯著。全廠機台效應洗刷乾淨後，各批次基線完美對齊，無明顯原料或環境硬傷。")
        
    print(f"\n{'='*90}\n")

print("="*90)
print("全線多軌交叉驗證完畢！請前往 Downloads 資料夾查收您的 CSV 成果報告。")
print("="*90)