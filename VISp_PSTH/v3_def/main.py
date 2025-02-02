import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from IPython.display import display
from pprint import pprint
import os
import sys
from sklearn.metrics import roc_auc_score

# brainbox / iblatlas / ONE 관련
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from brainbox.singlecell import bin_spikes
from iblatlas.atlas import AllenAtlas
from one.api import ONE

sys.path.append('VISp_PSTH/v3_def')
from print_cluster_info import print_cluster_info
from get_trial_masks import get_trial_masks
from compute_raster import compute_raster
from plot_raster import plot_raster
from plot_psth_2x2 import plot_psth_2x2
from plot_psth_contrast import plot_psth_contrast
from sub_func import save_file

# 현재 파일 위치로 이동
os.chdir(os.path.dirname(os.path.abspath(__file__)))
save_path = "../result/VISp"

# -----------------------------------------------------------------------------
# 1. ONE 초기화 및 세션 검색(유연한 brain_acronym)
# -----------------------------------------------------------------------------
brain_acronym = 'VISp'  # 예: 'VISp', 'MOs' 등

one = ONE()
sessions = one.search(atlas_acronym=brain_acronym, query_type='remote')
print(f"\n[{brain_acronym}] No. of detected sessions: {len(sessions)} \nExample 5 sessions:")
pprint(sessions[:5])

eid = 'ca4ecb4c-4b60-4723-9b9e-2c54a6290a53'
pid = one.eid2pid(eid)[0][0]

# -----------------------------------------------------------------------------
# 2. Trial / Spike data 로드 (SessionLoader, SpikeSortingLoader)
# -----------------------------------------------------------------------------
# Trial 데이터
sl = SessionLoader(eid=eid, one=one)
sl.load_trials()
events = sl.trials['stimOn_times'].values
left_idx, right_idx, correct_idx, incorrect_idx = get_trial_masks(sl.trials)  # trial mask

# Spike Sorting 데이터
ssl = SpikeSortingLoader(one=one, pid=pid, atlas=AllenAtlas())
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels)

# -----------------------------------------------------------------------------
# 3. 특정 영역(문자열 포함) Mask 정의 (모든 + good)
# -----------------------------------------------------------------------------
region_mask = np.array([brain_acronym in acr for acr in clusters['acronym']])
good_mask = (clusters['label'] == 1)

selected_mask_all = region_mask                # 모든 클러스터
selected_mask_good = region_mask & good_mask     # good 클러스터

clusters_good = {k: v[good_mask] for k, v in clusters.items()}
print_cluster_info(clusters, clusters_good, brain_acronym)

# -----------------------------------------------------------------------------
# 4. 기존 PSTH 및 raster plotting (생략; 기존 코드 그대로 실행)
# -----------------------------------------------------------------------------
for cluster_label, selected_mask in [("Single Clsuter", selected_mask_good),
                                      ("Good Clusters", selected_mask_good),
                                      ("All Clusters", selected_mask_all)]:
    
    selected_cluster_ids = np.where(selected_mask)[0]
    if cluster_label == "Single Clsuter":
        selected_cluster_ids = [selected_cluster_ids[0]]
    n_clusters = len(selected_cluster_ids)
    print(f"\n>> [{brain_acronym}] {cluster_label} in region: {n_clusters} clusters found.")
    
    if n_clusters == 0:
        print(f"No {cluster_label.lower()} in '{brain_acronym}' region. Skip.")
        continue
    
    spike_raster, times = compute_raster(spikes, selected_cluster_ids, events,
                                           pre_time=2, post_time=4, bin_size=0.001)
    
    plot_raster(spike_raster, times, events, brain_acronym, cluster_label, save_path)
    plot_psth_2x2(spike_raster, times, left_idx, right_idx, correct_idx, incorrect_idx,
                  brain_acronym, cluster_label, save_path)
    plot_psth_contrast(spike_raster, times, sl.trials, left_idx, right_idx, brain_acronym, cluster_label, save_path)

plt.show(block=False)
close = input("Close all figures? please press any key.")
plt.close('all')
print("Done.")

# -----------------------------------------------------------------------------
# 5. PSTH 배열, trial, neuron 정보 저장 (기존 코드)
# -----------------------------------------------------------------------------
spike_raster_save, times = compute_raster(spikes, np.where(selected_mask_good)[0],
                                            events, pre_time=2, post_time=4, bin_size=0.001)
# (Ntimebins X Ntrials X Nneurons)
spike_raster_save = spike_raster_save.transpose(2, 1, 0)
save_file(spike_raster_save, save_path="Z:\\ChaeHyeon_Seong", save_title="psth_Ntimebins_Ntrials_Nneurons")

trials = one.load_object(eid, 'trials')
trials_df = trials.to_df()
display(trials_df.head())
save_file(trials_df, save_path="Z:\\ChaeHyeon_Seong", save_title="trials_info")

clusters_df = clusters.to_df()
display(clusters_df.head())
save_file(clusters_df, save_path="Z:\\ChaeHyeon_Seong", save_title="neuron_info")

# -----------------------------------------------------------------------------
# 6. SP, DP, Mean Firing Rate 계산 및 Scatter Plot 그리기
# -----------------------------------------------------------------------------
# 아래 함수들은 각 뉴런별로 response window 내의 반응 여부를 기반으로 SP, DP, 그리고 mean firing rate을 계산합니다.

def compute_SP(spike_raster, times, trials_df, response_window=(0,0.2)):
    """
    각 뉴런별로 SP (Stimulus Probability)를 계산.
    - spike_raster: shape = [nNeurons, nTrials, nBins]
    - response_window: 자극 onset 후 반응을 측정할 시간 창 (초)
    - SP = (반응 확률 at contrast 100%) - (반응 확률 at contrast 0%)
    """
    idx_window = (times >= response_window[0]) & (times < response_window[1])
    # 좌/우 trial의 contrast 중 유효한 값(한쪽만 값이 있음)
    contrast = trials_df[['contrastLeft', 'contrastRight']].max(axis=1).values
    extreme_mask = (contrast == 0) | (contrast == 1)
    
    nNeurons = spike_raster.shape[0]
    sp_values = np.zeros(nNeurons)
    
    for neuron in range(nNeurons):
        counts = np.sum(spike_raster[neuron][:, idx_window], axis=1)
        responses = (counts > 0).astype(float)
        resp_extreme = responses[extreme_mask]
        contrast_extreme = contrast[extreme_mask]
        p0 = np.mean(resp_extreme[contrast_extreme == 0]) if np.any(contrast_extreme==0) else np.nan
        p1 = np.mean(resp_extreme[contrast_extreme == 1]) if np.any(contrast_extreme==1) else np.nan
        sp_values[neuron] = p1 - p0
    return sp_values

def compute_DP(spike_raster, times, trials_df, response_window=(0,0.2)):
    """
    각 뉴런별로 DP (Detect Probability)를 계산.
    - hit (feedbackType == 1)와 miss (feedbackType == -1) trial에서 response window 내 spike count를 비교.
    - ROC AUC를 DP로 산출.
    """
    idx_window = (times >= response_window[0]) & (times < response_window[1])
    feedback = trials_df['feedbackType'].values
    hit_mask = (feedback == 1)
    miss_mask = (feedback == -1)
    valid_mask = hit_mask | miss_mask
    
    nNeurons = spike_raster.shape[0]
    dp_values = np.zeros(nNeurons)
    
    for neuron in range(nNeurons):
        counts = np.sum(spike_raster[neuron][:, idx_window], axis=1)
        counts_valid = counts[valid_mask]
        labels = feedback[valid_mask]
        labels_binary = (labels == 1).astype(int)
        if len(np.unique(labels_binary)) < 2:
            dp_values[neuron] = np.nan
        else:
            dp_values[neuron] = roc_auc_score(labels_binary, counts_valid)
    return dp_values

def compute_mean_FR(spike_raster, times, window=(0,0.2)):
    """
    각 뉴런별로 mean firing rate (Hz)를 계산.
    - window 내 spike count / window duration의 trial 평균.
    """
    idx_window = (times >= window[0]) & (times < window[1])
    nNeurons = spike_raster.shape[0]
    mean_FR = np.zeros(nNeurons)
    
    for neuron in range(nNeurons):
        fr_trials = np.sum(spike_raster[neuron][:, idx_window], axis=1) / (window[1]-window[0])
        mean_FR[neuron] = np.mean(fr_trials)
    return mean_FR

# spike_raster_neurons: [nNeurons, nTrials, nBins]
# 여기서는 good clusters에 해당하는 뉴런들만 사용
neuron_ids = np.where(selected_mask_good)[0]
spike_raster_neurons, times = compute_raster(spikes, neuron_ids, events, pre_time=2, post_time=4, bin_size=0.001)

# trials 정보를 DataFrame 형태로 (이미 trials_df가 있으므로 사용)
trials_df = sl.trials.to_df()

# SP, DP, mean FR 계산 (response window: 0~0.2초)
sp_values = compute_SP(spike_raster_neurons, times, trials_df, response_window=(0,0.2))
dp_values = compute_DP(spike_raster_neurons, times, trials_df, response_window=(0,0.2))
mean_FR = compute_mean_FR(spike_raster_neurons, times, window=(0,0.2))

# Scatter plot: Mean Firing Rate vs SP
plt.figure(figsize=(6,5))
plt.scatter(mean_FR, sp_values, c='royalblue', edgecolor='k')
plt.xlabel('Mean Firing Rate (Hz)')
plt.ylabel('Stimulus Probability (SP)')
plt.title('Mean Firing Rate vs SP')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join("Z:\\ChaeHyeon_Seong", "SP_vs_MeanFiringRate.png"), dpi=300)
plt.show()

# Scatter plot: Mean Firing Rate vs DP
plt.figure(figsize=(6,5))
plt.scatter(mean_FR, dp_values, c='darkorange', edgecolor='k')
plt.xlabel('Mean Firing Rate (Hz)')
plt.ylabel('Detect Probability (DP, ROC AUC)')
plt.title('Mean Firing Rate vs DP')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join("Z:\\ChaeHyeon_Seong", "DP_vs_MeanFiringRate.png"), dpi=300)
plt.show()
