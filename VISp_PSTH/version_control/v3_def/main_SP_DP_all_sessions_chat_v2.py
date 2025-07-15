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

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# -----------------------------------------------------------------------------
# 1. AUROC 및 ROC Curve 계산 함수 (Mean ROC 추가)
# -----------------------------------------------------------------------------
from compute_sp import compute_sp
from compute_dp import compute_dp, compute_dp_contrast
from plot_sp import plot_sp, plot_sp_transformed, plot_sp_roc
from plot_dp import plot_dp, plot_dp_contrast, plot_dp_transformed, plot_dp_contrast_transformed, plot_dp_roc

brain_acronym = 'VISp'
one = ONE()

sessions = one.search(atlas_acronym=brain_acronym, query_type='remote')
sessions = list(sessions)
print(f"Found {len(sessions)} sessions for region: {brain_acronym}")
print(f"Found {len(sessions)} sessions for region: {brain_acronym}")

data_path = "C:/Users/miasc/SCH/shinlab/ChaeHyeon_Seong/dataset_v2"
trials_df_all = pd.DataFrame()

# session_summary_list를 만들어 각 세션의 task type 등 정보를 저장
session_summary_list = []

i = 1
for eid in sessions:
    if i == 3:
        break
    print(f"\n=== [Session{i}: {eid}] ===")
    i += 1
    session_folder = os.path.join(data_path, eid)
    
    # trials_info.csv 파일 로드
    trials_info_path = os.path.join(session_folder, "trials_info.csv")
    trials_info = pd.read_csv(trials_info_path)
    
    # trials_info 파일에 task_type 열이 있다면, 해당 세션의 task_type 결정
    if 'task_type' in trials_info.columns:
        unique_tasks = trials_info['task_type'].dropna().unique()
        if len(unique_tasks) == 1:
            task_type = unique_tasks[0]
        else:
            task_type = ','.join(unique_tasks.astype(str))
    else:
        task_type = 'unknown'
        
    # 추가 정보: 시행 수
    n_trials = len(trials_info)
    
    # session_summary_list에 세션 정보를 저장 (session id, task type, 시행 수 등)
    session_summary_list.append({
        'session_id': eid,
        'task_type': task_type,
        'n_trials': n_trials
    })
    
    # trials 정보를 전체 DataFrame에 추가 (필요시)
    trials_df_all = pd.concat([trials_df_all, trials_info], ignore_index=True)

# 최종적으로 세션 요약 테이블 생성 후 출력
session_summary_df = pd.DataFrame(session_summary_list)
print("\nSession Summary:")
display(session_summary_df)

reaction_time = trials_df_all['response_times'] - trials_df_all['stimOn_times']
print(f'Average reaction time: {np.mean(reaction_time)}')
left_reaction_time = np.mean(reaction_time[~np.isnan(trials_df_all['contrastLeft'])])
right_reaction_time = np.mean(reaction_time[~np.isnan(trials_df_all['contrastRight'])])
print(f'Average reaction time for stimulus on')
print(f'Left: {left_reaction_time}')
print(f'Right: {right_reaction_time}')

# 원래 trial PSTH 윈도우용 설정 (Method 2, 3)
pre_time = 2.0
post_time = 4.0
bin_size = 0.001
times = np.arange(-pre_time, post_time, bin_size)
start_time, end_time = 0, 0.1 # 0 ms ~ 100 ms
start_idx = np.searchsorted(times, start_time)
end_idx = np.searchsorted(times, end_time)

contrast_levels = [0.0625, 0.125, 0.25]
colors = ['r', 'g', 'b']

# 각 방식에 따른 FR 값을 저장할 리스트 (Method 1: session 전체, Method 2: 프리스티뮬러스, Method 3: catch trials)
mean_fr_session_all = []   # Method 1: 전체 recording FR (Code 1에서 저장한 파일 사용)
mean_fr_prestim_all = []   # Method 2: 프리스티뮬러스 구간 평균 FR
mean_fr_catch_all = []     # Method 3: catch trial 평균 FR

sp_values_all = []
dp_contrast_values_all = {contrast: [] for contrast in [0.0625, 0.125, 0.25]}
dp_values_all = []

i = 1
for eid in sessions:
    if i == 3:
        break
    print(f"\n=== [Session{i}: {eid}] ===")
    i += 1
    session_folder = os.path.join(data_path, eid)
    trials_df = pd.read_csv(os.path.join(session_folder, "trials_info.csv"))

    pids, labels = one.eid2pid(eid)

    for pid, label in zip(pids, labels):
        print(f"   -> Probe: {pid} ({label})")
        probe_folder = os.path.join(session_folder, label)

        # PSTH 데이터 로드 (trial-locked)
        psth = np.load(os.path.join(probe_folder, "psth.npy"))

        # Method 1: 전체 session FR
        mean_fr_session = np.load(os.path.join(probe_folder, "mean_fr_session.npy"))
        mean_fr_session_all.append(mean_fr_session)

        # Method 2: 프리스티뮬러스 구간 (-0.6 ~ -0.1초)
        pre_window_start = -0.6
        pre_window_end = -0.1
        pre_start_idx = np.searchsorted(times, pre_window_start)
        pre_end_idx = np.searchsorted(times, pre_window_end)
        mean_fr_prestim = np.mean(psth[:, :, pre_start_idx:pre_end_idx], axis=(1, 2))
        mean_fr_prestim_all.append(mean_fr_prestim)

        # Method 3: Catch trials (자극 없는 trial)
        catch_mask = trials_df['contrastLeft'].isna() & trials_df['contrastRight'].isna()
        if np.sum(catch_mask) > 0:
            mean_fr_catch = np.mean(psth[:, catch_mask, :], axis=(1, 2))
        else:
            mean_fr_catch = np.full(psth.shape[0], np.nan)
        mean_fr_catch_all.append(mean_fr_catch)

        # SP 및 DP 계산

        sp_values, _ = compute_sp(psth, trials_df, start_idx, end_idx)
        sp_values_all.append(sp_values)

        contrast = [0.0625, 0.125, 0.25]
        dp_values, _ = compute_dp(psth, trials_df, contrast, start_idx, end_idx)
        dp_values_all.append(dp_values)

        for contrast in [0.0625, 0.125, 0.25]:
            mask_contrast = (trials_df['contrastRight'] == contrast)
            dp_values, _ = compute_dp_contrast(psth, trials_df, mask_contrast, start_idx, end_idx)
            dp_contrast_values_all[contrast].append(dp_values)

mean_fr_session_all = np.concatenate(mean_fr_session_all)
mean_fr_prestim_all = np.concatenate(mean_fr_prestim_all)
mean_fr_catch_all = np.concatenate(mean_fr_catch_all)
sp_values_all = np.concatenate(sp_values_all)
dp_values_all = np.concatenate(dp_values_all)
dp_contrast_values_all = {contrast: np.concatenate(dp_contrast_values_all[contrast])
                            for contrast in dp_contrast_values_all}

print(f"Total mean firing rates (session full recording): {mean_fr_session_all.shape}")
print(f"Total mean firing rates (pre-stimulus): {mean_fr_prestim_all.shape}")
print(f"Total mean firing rates (catch trials): {mean_fr_catch_all.shape}")
# 만약 mean_fr_session_all.shape이 나머지 값과 다르다면,
# 세션마다 다른 길이의 PSTH를 가지고 있을 수 있음
# mean_fr_session_all.shape을 나머지 값만큼 줄이기
mean_fr_session_all = mean_fr_session_all[:len(mean_fr_prestim_all)]

print(f"Total SP values: {sp_values_all.shape}")
print(f"Total DP values: {dp_values_all.shape}")
for contrast, dp_values in dp_contrast_values_all.items(): 
    print(f"Total DP values for contrast {contrast*100}%: {dp_values.shape}")

cluster_type = 'good'
save_path_out = f"../result/VISp/SP_DP/all_sessions/v2/SO+{round(end_time, 2)}s/{cluster_type}_clusters"

save_file(mean_fr_session_all, save_path=save_path_out, save_title="mean_fr_session_all")
save_file(mean_fr_prestim_all, save_path=save_path_out, save_title="mean_fr_prestim_all")
save_file(mean_fr_catch_all, save_path=save_path_out, save_title="mean_fr_catch_all")
save_file(sp_values_all, save_path=save_path_out, save_title="sp_values_all")
save_file(dp_values_all, save_path=save_path_out, save_title="dp_values_all")
save_file(dp_contrast_values_all, save_path=save_path_out, save_title="dp_contrast_values_all")

plot_info_sp = [brain_acronym, cluster_type, round(end_time, 2)]
plot_info_dp = [brain_acronym, cluster_type, round(end_time, 2)]
plot_info_dp_contrast = [brain_acronym, cluster_type, round(end_time, 2), contrast_levels, colors]

plot_sp(sp_values_all, mean_fr_session_all, plot_info_sp, save_path_out)
plot_dp(dp_values_all, mean_fr_session_all, plot_info_dp, save_path_out)
plot_dp_contrast(dp_contrast_values_all, mean_fr_session_all, plot_info_dp_contrast, save_path_out)

plot_sp_transformed(sp_values_all, mean_fr_session_all, plot_info_sp, save_path_out)
plot_dp_transformed(dp_values_all, mean_fr_session_all, plot_info_dp, save_path_out)
plot_dp_contrast_transformed(dp_contrast_values_all, mean_fr_session_all, plot_info_dp_contrast, save_path_out)



# mean fr 세 방법끼리 비교하는 그래프 - Method 1, 2, 3 subplot
plt.figure(figsize=(10, 6))
plt.subplot(3, 1, 1)
plt.plot(mean_fr_session_all, label='Method 1: 전체 recording FR')
plt.xlabel('Neuron Index')
plt.ylabel('Firing Rate (Hz)')
plt.title('Method 1: 전체 recording FR')
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(mean_fr_prestim_all, label='Method 2: Pre-stimulus FR')
plt.xlabel('Neuron Index')
plt.ylabel('Firing Rate (Hz)')
plt.title('Method 2: Pre-stimulus FR')
plt.grid()


plt.subplot(3, 1, 3)
plt.plot(mean_fr_catch_all, label='Method 3: Catch Trials FR')
plt.xlabel('Neuron Index')
plt.ylabel('Firing Rate (Hz)')
plt.title('Method 3: Catch Trials FR')
plt.grid()
plt.tight_layout()
plt.savefig(os.path.join(save_path_out, "mean_fr_comparison.png"))

plt.show()
input("Press any key to close the figure...\n")
plt.close()
