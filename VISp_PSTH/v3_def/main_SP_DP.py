import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from IPython.display import display
from pprint import pprint
import os
import sys

# brainbox / iblatlas / ONE 관련
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from brainbox.singlecell import bin_spikes
from iblatlas.atlas import AllenAtlas
from one.api import ONE

import sys
sys.path.append('VISp_PSTH/v3_def')
from print_cluster_info import print_cluster_info
from get_trial_masks import get_trial_masks
from compute_raster import compute_raster
from plot_raster import plot_raster
from plot_psth_2x2 import plot_psth_2x2
from plot_psth_contrast import plot_psth_contrast
from compute_SP import compute_SP
from compute_DP import compute_DP
from compute_mean_FR import compute_mean_FR
from sub_func import save_file

# 현재 파일 위치로 이동
os.chdir(os.path.dirname(os.path.abspath(__file__)))
save_path = "../result/VISp"

# -----------------------------------------------------------------------------
# 1. ONE 초기화 및 세션 검색(유연한 brain_acronym)
# -----------------------------------------------------------------------------

brain_acronym = 'VISp' # 원하는 뇌영역 문자열을 지정(예: 'VISp', 'MOs' 등)

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
left_idx, right_idx, correct_idx, incorrect_idx = get_trial_masks(sl.trials) # trial mask

# Spike Sorting 데이터
ssl = SpikeSortingLoader(one=one, pid=pid, atlas=AllenAtlas())
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels)

# -----------------------------------------------------------------------------
# 3. 특정 영역(문자열 포함) Mask 정의 (모든 + good)
# -----------------------------------------------------------------------------
region_mask = np.array([brain_acronym in acr for acr in clusters['acronym']]) # ex) 'VISp1', 'VISp2', ...
good_mask = (clusters['label'] == 1)

selected_mask_all = region_mask  # all clusters in region (ex. VISp)
selected_mask_good = region_mask & good_mask  # good clusters in region (ex. VISp)

clusters_good = {k: v[good_mask] for k, v in clusters.items()} # Good clusters만 추출

print_cluster_info(clusters, clusters_good, brain_acronym)


# === PSTH 및 Trials 데이터 로드 후, SP/DP 계산 및 시각화 ===

# 예: PSTH 계산
spike_raster, times = compute_raster(spikes, selected_mask_good, events,
                                       pre_time=2, post_time=4, bin_size=0.001)

# SP, DP 계산 (response window: 자극 onset 후 0~0.2초로 설정)
sp_values = compute_SP(spike_raster, times, sl.trials, response_window=(0, 0.2))
dp_values = compute_DP(spike_raster, times, sl.trials, response_window=(0, 0.2))

print("SP for each neuron:", sp_values)
print("DP for each neuron:", dp_values)

# Mean firing rate 계산 (예: 0~0.2초 window 사용)
mean_FR = compute_mean_FR(spike_raster, times, window=(0, 0.2))

# --- Scatter Plot 그리기 ---
# 1) Mean Firing Rate vs. SP
plt.figure(figsize=(6, 5))
plt.scatter(mean_FR, sp_values, c='royalblue', edgecolor='k')
plt.xlabel('Mean Firing Rate (Hz)')
plt.ylabel('SP (p_resp(100%) - p_resp(0%))')
plt.title('SP vs. Mean Firing Rate')
plt.grid(True)
plt.tight_layout()
plt.savefig("SP_vs_MeanFiringRate.png", dpi=300)
plt.show()

# 2) Mean Firing Rate vs. DP
plt.figure(figsize=(6, 5))
plt.scatter(mean_FR, dp_values, c='darkorange', edgecolor='k')
plt.xlabel('Mean Firing Rate (Hz)')
plt.ylabel('DP (ROC AUC)')
plt.title('DP vs. Mean Firing Rate')
plt.grid(True)
plt.tight_layout()
plt.savefig("DP_vs_MeanFiringRate.png", dpi=300)
plt.show()
