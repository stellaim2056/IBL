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
from compute_dp import compute_dp

from plot_sp import plot_sp, plot_sp_transformed, plot_sp_roc
from plot_dp import plot_dp, plot_dp_transformed, plot_dp_roc



# -----------------------------------------------------------------------------
# 2. 데이터 로드 및 전처리
# -----------------------------------------------------------------------------
brain_acronym = 'VISp'
one = ONE()

sessions = one.search(atlas_acronym=brain_acronym, query_type='remote')
print(f"Found {len(sessions)} sessions for region: {brain_acronym}")

data_path = "Z:\ChaeHyeon_Seong"
trials_df = pd.DataFrame()

for eid in sessions:
    print(f"\n=== [Session: {eid}] ===")
    session_folder = os.path.join(data_path, eid)
    trials_info = pd.read_csv(os.path.join(session_folder, "trials_info.csv"))
    trials_df = pd.concat([trials_df, trials_info], ignore_index=True)

# -----------------------------------------------------------------------------
# reaction_time 계산
# -----------------------------------------------------------------------------
reaction_time = trials_df['response_times'] - trials_df['stimOn_times']
print(f'Average reaction time: {np.mean(reaction_time)}')
left_reaction_time = np.mean(reaction_time[~np.isnan(trials_df['contrastLeft'])])
right_reaction_time = np.mean(reaction_time[~np.isnan(trials_df['contrastRight'])])
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

# -----------------------------------------------------------------------------
clusters = pd.DataFrame()
mean_fr_values = np.array([])
sp_values = np.array([])
dp_values_dict = {}

for eid in sessions:
    print(f"\n=== [Session: {eid}] ===")
    session_folder = os.path.join(data_path, eid)
    trials_info = pd.read_csv(os.path.join(session_folder, "trials_info.csv"))
    trials_df = pd.concat([trials_df, trials_info], ignore_index=True)

    pids, labels = one.eid2pid(eid)

    for pid, label in zip(pids, labels):
        print(f"   -> Probe: {pid} ({label})")
        probe_folder = os.path.join(session_folder, label)

        neuron_info = pd.read_csv(os.path.join(probe_folder, "neuron_info.csv"))

        clusters = pd.concat([clusters, neuron_info], ignore_index=True)
        psth = np.load(os.path.join(probe_folder, "psth.npy"))

        # -----------------------------------------------------------------------------
        # 3. SP & DP 계산
        # -----------------------------------------------------------------------------
        # mean firing rate 계산
        mean_fr_values = np.mean(psth[:, :, start_idx:end_idx], axis=(1,2)) # shape = (n_neurons,)

        # SP 계산
        sp_values, mean_roc_sp = compute_sp(psth, trials_df, start_idx, end_idx) # sp_values.shape = (n_neurons,)

        # DP 계산
        contrast_levels = [0.0625, 0.125, 0.25]
        colors = ['r', 'g', 'b']  # Contrast 별 색상

        dp_values_dict = {}
        mean_roc_dp_dict = {}

        for contrast, color in zip(contrast_levels, colors):
            mask_contrast = (trials_df['contrastRight'] == contrast)

            dp_values, mean_roc_dp = compute_dp(psth, trials_df, mask_contrast, start_idx, end_idx)

            dp_values_dict[contrast] = dp_values
            mean_roc_dp_dict[contrast] = mean_roc_dp

    # -----------------------------------------------------------------------------
    # data shape 정보
    # -----------------------------------------------------------------------------
    print(f"Trials shape: {trials_df.shape}")
    print(f"Clusters shape: {clusters.shape}")
    print(f"PSTHs shape: {psth.shape}")
    print(f"Time bins shape: {times.shape}")

# column 제목들만 추출
print('Keys of trials:', list(trials_df.keys()))
display(trials_df.head())



# -----------------------------------------------------------------------------
# 4. 그래프 시각화
# -----------------------------------------------------------------------------
cluster_type = 'good'  # 'all' or 'good'

save_path = f"../result/VISp/SP_DP/all_sessions/SO+{end_time}s/{cluster_type}_clusters"
plot_info_sp = [brain_acronym, cluster_type, end_time]
plot_info_dp = [brain_acronym, cluster_type, end_time, contrast_levels, colors]

plot_sp(sp_values, mean_fr_values, plot_info_sp, save_path)
plot_dp(dp_values_dict, mean_fr_values, plot_info_dp, save_path)

# -----------------------------------------------------------------------------
# 4. 그래프 시각화 (원본 vs 변환)
# -----------------------------------------------------------------------------
plot_sp_transformed(sp_values, mean_fr_values, plot_info_sp, save_path)
plot_dp_transformed(dp_values_dict, mean_fr_values, plot_info_dp, save_path)

# -----------------------------------------------------------------------------
# 4. 그래프 시각화 (1x2)
# -----------------------------------------------------------------------------
plot_sp_roc(sp_values, mean_roc_sp, mean_fr_values, plot_info_sp, save_path)
plot_dp_roc(dp_values_dict, mean_roc_dp_dict, mean_fr_values, plot_info_dp, save_path)

plt.show()

# 아무 키나 누르면 피규어 창이 닫힙니다.
input("Press any key to close the figure...\n")
plt.close()

