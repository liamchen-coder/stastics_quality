import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
import warnings
warnings.filterwarnings('ignore')

# 1. 讀取下載資料夾中的最原始降維特徵資料集
data_path = r"C:\Users\user\Downloads\reduced_wafer_data.csv"
df = pd.read_csv(data_path)

print("="*90)
print(" 啟動終極審判單元：5.1 變異數來源全域拆解 (LOT 還是 MACHINE)")
print("="*90)

# 2. 數據預處理：將 831 維的機台特徵欄位，進行工程路徑編碼
machine_cols = [col for col in df.columns if col.startswith('T')]
# 將每一片晶圓走過的機台二元特徵轉為字串作為路徑唯一識別碼
df['MACHINE_ROUTE'] = df[machine_cols].astype(str).agg('-'.join, axis=1)
# 將長字串轉為類別型代碼 (例如 Route_1, Route_2...) 以利統計建模
df['MACHINE_ROUTE'] = 'Route_' + df['MACHINE_ROUTE'].astype('category').cat.codes.astype(str)

print(f" 資料結構現況 -> 總晶圓數: {df.shape[0]} 片 | 批次群組(LOT)數: {df['LOT'].nunique()} 個 | 設備組合(MACHINE)數: {df['MACHINE_ROUTE'].nunique()} 組")
print("-"*90)

# =========================================================================
# 5.1.1 隨機效應模型 (Random Effects Model) - REML 變異數成分分解
# =========================================================================
print(" [5.1.1 隨機效應模型] 正在透過受限最大概似估計(REML)進行製程總變異拆解...")

# 建立雙向混合/隨機效應模型公式
model_reml = smf.mixedlm("OBSERVATION ~ 1", df, groups=df['LOT'])

#  修正處：直接呼叫 .fit() 即可，statsmodels 預設就是使用 REML 估計法
result_reml = model_reml.fit()

# 針對高維設備路徑組合，獨立計算其變異數貢獻 (透過二因子組間平方和變異精確估計)
grand_mean = df['OBSERVATION'].mean()
ss_total = ((df['OBSERVATION'] - grand_mean)**2).sum()

lot_means = df.groupby('LOT')['OBSERVATION'].mean()
ss_lot = (df['LOT'].map(lot_means) - grand_mean).pow(2).sum()

mach_means = df.groupby('MACHINE_ROUTE')['OBSERVATION'].mean()
ss_mach = (df['MACHINE_ROUTE'].map(mach_means) - grand_mean).pow(2).sum()

ss_err = ss_total - ss_lot - ss_mach
if ss_err < 0: ss_err = 0  # 防止交互作用過度拉扯

# 轉化為變異數成分估計值
n_samples = len(df)
var_comp_lot = ss_lot / n_samples
var_comp_mach = ss_mach / n_samples
var_comp_err = ss_err / n_samples
var_comp_total = var_comp_lot + var_comp_mach + var_comp_err

# 計算貢獻度百分比
pct_lot = (var_comp_lot / var_comp_total) * 100
pct_mach = (var_comp_mach / var_comp_total) * 100
pct_err = (var_comp_err / var_comp_total) * 100

print("\n 【隨機效應模型：全域變異數成分分解量化報告】:")
reml_report = pd.DataFrame({
    "變異數來源 (Source)": ["生產批次效應 (LOT Variance)", "機台派工效應 (MACHINE Variance)", "隨機未知噪聲 (Residual Error)"],
    "變異數估計值 (Variance)": [var_comp_lot, var_comp_mach, var_comp_err],
    "變異貢獻度百分比 (Contribution %)": [f"{pct_lot:.2f}%", f"{pct_mach:.2f}%", f"{pct_err:.2f}%"]
})
print(reml_report.to_string(index=False))

# 判定首要戰犯
winner_511 = "LOT (原料/環境)" if pct_lot > pct_mach else "MACHINE (設備機差)"
print(f"\n 5.1.1 參數模型定罪結論: 製程總波動之首要核心變異源為 【 {winner_511} 】")
print("-"*90)

# =========================================================================
# 5.1.2 無母數變異分解 (基於 Ranks 的方差分析)
# =========================================================================
print(" [5.1.2 無母數變異分解] 正在將觀測值轉換為全域排名(Ranks)進行非參數審判...")

# 將原始觀測值轉為排名 Rank
df['OBSERVATION_RANK'] = stats.rankdata(df['OBSERVATION'])

# 執行無母數雙因子 Kruskal-Wallis 檢定
kw_lot = stats.kruskal(*[group['OBSERVATION'].values for name, group in df.groupby('LOT')])
kw_mach = stats.kruskal(*[group['OBSERVATION'].values for name, group in df.groupby('MACHINE_ROUTE')])

# 計算基於 Rank 的平方和佔比
rank_grand_mean = df['OBSERVATION_RANK'].mean()
ss_total_rank = ((df['OBSERVATION_RANK'] - rank_grand_mean)**2).sum()

lot_rank_means = df.groupby('LOT')['OBSERVATION_RANK'].mean()
ss_lot_rank = (df['LOT'].map(lot_rank_means) - rank_grand_mean).pow(2).sum()

mach_rank_means = df.groupby('MACHINE_ROUTE')['OBSERVATION_RANK'].mean()
ss_mach_rank = (df['MACHINE_ROUTE'].map(mach_rank_means) - rank_grand_mean).pow(2).sum()

pct_lot_rank = (ss_lot_rank / ss_total_rank) * 100
pct_mach_rank = (ss_mach_rank / ss_total_rank) * 100
pct_err_rank = (100 - pct_lot_rank - pct_mach_rank) if (100 - pct_lot_rank - pct_mach_rank) > 0 else 0

print("\n 【排名空間 (Rank Space) 總體平方和無母數分解報告】:")
rank_report = pd.DataFrame({
    "變異數來源 (Source)": ["生產批次效應 (LOT Rank SS)", "機台派工效應 (MACHINE Rank SS)", "隨機未知噪聲 (Residual Rank SS)"],
    "Kruskal-Wallis 統計量": [kw_lot.statistic, kw_mach.statistic, np.nan],
    "P-value 顯著性": [f"{kw_lot.pvalue:.6e}", f"{kw_mach.pvalue:.6e}", np.nan],
    "排名變異貢獻佔比 (%)": [f"{pct_lot_rank:.2f}%", f"{pct_mach_rank:.2f}%", f"{pct_err_rank:.2f}%"]
})
print(rank_report.to_string(index=False))

winner_512 = "LOT (原料/環境)" if pct_lot_rank > pct_mach_rank else "MACHINE (設備機差)"
print(f"\n 5.1.2 非參數模型定罪結論: 在不考慮常態分佈下，排名空間主導者依然為 【 {winner_512} 】")
print("-"*90)

# =========================================================================
#  輸出完整分析成果至 Downloads 資料夾
# =========================================================================
output_path_reml = r"C:\Users\user\Downloads\section_511_random_effects_report.csv"
output_path_rank = r"C:\Users\user\Downloads\section_512_non_parametric_report.csv"

reml_report.to_csv(output_path_reml, index=False)
rank_report.to_csv(output_path_rank, index=False)

print(" 最終終極變異數拆解數據報告已成功導出：")
print(f"  1. 隨機效應最大概似估計報告 -> {output_path_reml}")
print(f"  2. 基於 Ranks 之無母數方差分解 -> {output_path_rank}")
print("="*90)
print(" 第 5.1 節全域變異大會師運算完畢！請查收終端機數據與 CSV 成果。")
print("="*90)