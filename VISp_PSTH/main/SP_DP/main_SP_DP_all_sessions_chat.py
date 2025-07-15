import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.interpolate import interp1d
from IPython.display import display
import pandas as pd
import os
import sys

# brainbox / iblatlas / ONE 관련
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from iblatlas.atlas import AllenAtlas
from one.api import ONE

sys.path.append('VISp_PSTH/v3_def')
from compute_raster import compute_raster
from sub_func import save_file

# 현재 파일 위치로 이동
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------------
# 1. AUROC 및 ROC Curve 계산 함수 (Mean ROC 추가)
# -----------------------------------------------------------------------------
from compute_sp import compute_sp
from compute_dp import compute_dp, compute_dp_contrast

from plot_sp import plot_sp, plot_sp_transformed, plot_sp_roc
from plot_dp import plot_dp, plot_dp_contrast, plot_dp_transformed, plot_dp_contrast_transformed, plot_dp_roc



# -----------------------------------------------------------------------------
# 2. 데이터 로드 및 전처리
# -----------------------------------------------------------------------------
brain_acronym = 'VISp'
one = ONE()

sessions = one.search(atlas_acronym=brain_acronym, query_type='remote')
sessions = list(sessions)
print(f"Found {len(sessions)} sessions for region: {brain_acronym}")
# for i in [44, 45, 68, 68, 68]:
#     sessions.pop(i)
print(f"Found {len(sessions)} sessions for region: {brain_acronym}")

data_path = "C:/Users/miasc/SCH/shinlab/ChaeHyeon_Seong"
trials_df_all = pd.DataFrame()

i=1
for eid in sessions:
    if i == 6:
        break
    print(f"\n=== [Session{i}: {eid}] ==="); i+=1
    session_folder = data_path + "/" + eid # os.path.join(data_path, eid)
    trials_info = pd.read_csv(session_folder + "/trials_info.csv") # os.path.join(session_folder, "trials_info.csv")
    trials_df_all = pd.concat([trials_df_all, trials_info], ignore_index=True)

# -----------------------------------------------------------------------------
# reaction_time 계산
# -----------------------------------------------------------------------------
reaction_time = trials_df_all['response_times'] - trials_df_all['stimOn_times']
print(f'Average reaction time: {np.mean(reaction_time)}')
left_reaction_time = np.mean(reaction_time[~np.isnan(trials_df_all['contrastLeft'])])
right_reaction_time = np.mean(reaction_time[~np.isnan(trials_df_all['contrastRight'])])
print(f'Average reaction time for stimulus on')
print(f'Left: {left_reaction_time}')
print(f'Right: {right_reaction_time}')

pre_time = 2.0
post_time = 4.0
bin_size = 0.001
times = np.arange(-pre_time, post_time, bin_size)
start_time, end_time = 0, right_reaction_time
start_idx = np.searchsorted(times, start_time)
end_idx = np.searchsorted(times, end_time)

contrast_levels = [0.0625, 0.125, 0.25]
colors = ['r', 'g', 'b']  # Contrast 별 색상

# -----------------------------------------------------------------------------
clusters = pd.DataFrame()
mean_fr_values_all = []
sp_values_all = []
dp_contrast_values_all = {contrast: [] for contrast in [0.0625, 0.125, 0.25]}
dp_values_all = []

i=1
for eid in sessions:
    if i == 6:
        break
    print(f"\n=== [Session{i}: {eid}] ==="); i+=1
    session_folder = os.path.join(data_path, eid)
    trials_df = pd.read_csv(os.path.join(session_folder, "trials_info.csv"))

    pids, labels = one.eid2pid(eid)

    for pid, label in zip(pids, labels):
        print(f"   -> Probe: {pid} ({label})")
        probe_folder = os.path.join(session_folder, label)

        # neuron_info = pd.read_csv(os.path.join(probe_folder, "neuron_info.csv"))
        psth = np.load(os.path.join(probe_folder, "psth.npy"))

        # mean firing rate 계산
        mean_fr_values = np.mean(psth[:, :, start_idx:end_idx], axis=(1, 2))
        mean_fr_values_all.append(mean_fr_values)

        # SP 계산
        sp_values, _ = compute_sp(psth, trials_df, start_idx, end_idx)
        sp_values_all.append(sp_values)

        # DP 계산 (contrast 상관없이)
        contrast = [0.0625, 0.125, 0.25]
        dp_values, _ = compute_dp(psth, trials_df, contrast, start_idx, end_idx)
        dp_values_all.append(dp_values)

        # DP 계산 (contrast별)
        for contrast in [0.0625, 0.125, 0.25]:
            mask_contrast = (trials_df['contrastRight'] == contrast)
            dp_values, _ = compute_dp_contrast(psth, trials_df, mask_contrast, start_idx, end_idx)
            dp_contrast_values_all[contrast].append(dp_values)

        

# ----------------------------------------------------------------------------- 
# 모든 세션의 결과를 하나로 결합 
# ----------------------------------------------------------------------------- 
# data_path = f"../result/VISp/SP_DP/all_sessions/SO+{end_time}s/good_clusters"
# mean_fr_values_all = np.load(os.path.join(data_path, "mean_fr_values_all.npy"))
# sp_values_all = np.load(os.path.join(data_path, "sp_values_all.npy"))
# dp_contrast_values_all = np.load(os.path.join(data_path, "dp_contrast_values_all.npy"), allow_pickle=True).item()


mean_fr_values_all = np.concatenate(mean_fr_values_all)
sp_values_all = np.concatenate(sp_values_all)
dp_values_all = np.concatenate(dp_values_all)
dp_contrast_values_all = {contrast: np.concatenate(dp_contrast_values_all[contrast]) for contrast in dp_contrast_values_all}



# 확인용 출력
print(f"Total mean firing rates: {mean_fr_values_all.shape}")
print(f"Total SP values: {sp_values_all.shape}")
print(f"Total DP values: {dp_values_all.shape}")
for contrast, dp_values in dp_contrast_values_all.items(): 
    print(f"Total DP values for contrast {contrast*100}%: {dp_values.shape}")


# # column 제목들만 추출
# print('Keys of trials:', list(trials_df.keys()))
# display(trials_df.head())



# -----------------------------------------------------------------------------
# 4. 그래프 시각화
# -----------------------------------------------------------------------------
cluster_type = 'good'  # 'all' or 'good'


save_path = f"../result/VISp/SP_DP/all_sessions/SO+{round(end_time, 2)}s/{cluster_type}_clusters"

save_file(mean_fr_values_all, save_path=save_path, save_title="mean_fr_values_all")
save_file(sp_values_all, save_path=save_path, save_title="sp_values_all")
save_file(dp_values_all, save_path=save_path, save_title="dp_values_all")
save_file(dp_contrast_values_all, save_path=save_path, save_title="dp_contrast_values_all")


plot_info_sp = [brain_acronym, cluster_type, round(end_time, 2)]
plot_info_dp = [brain_acronym, cluster_type, round(end_time, 2)]
plot_info_dp_contrast = [brain_acronym, cluster_type, round(end_time, 2), contrast_levels, colors]


plot_sp(sp_values_all, mean_fr_values_all, plot_info_sp, save_path)
plot_dp(dp_values_all, mean_fr_values_all, plot_info_dp, save_path)
plot_dp_contrast(dp_contrast_values_all, mean_fr_values_all, plot_info_dp_contrast, save_path)



# -----------------------------------------------------------------------------
# 4. 그래프 시각화 (원본 vs 변환)
# -----------------------------------------------------------------------------
plot_sp_transformed(sp_values_all, mean_fr_values_all, plot_info_sp, save_path)
plot_dp_transformed(dp_values_all, mean_fr_values_all, plot_info_dp, save_path)
plot_dp_contrast_transformed(dp_contrast_values_all, mean_fr_values_all, plot_info_dp_contrast, save_path)


# -----------------------------------------------------------------------------
# 4. 그래프 시각화 (1x2)
# -----------------------------------------------------------------------------
# plot_sp_roc(sp_values, mean_roc_sp, mean_fr_values, plot_info_sp, save_path)
# plot_dp_roc(dp_values_dict, mean_roc_dp_dict, mean_fr_values, plot_info_dp, save_path)

plt.show()

# 아무 키나 누르면 피규어 창이 닫힙니다.
input("Press any key to close the figure...\n")
plt.close()

