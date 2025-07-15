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

# -----------------------------------------------------------------------------
# 6. 두 가지 경우(모든 클러스터 / good 클러스터)에 대해 동일한 분석
# -----------------------------------------------------------------------------
for cluster_label, selected_mask in [("Single Clsuter", selected_mask_good),
                                ("Good Clusters", selected_mask_good),
                                 ("All Clusters", selected_mask_all)]:
    
    selected_cluster_ids = np.where(selected_mask)[0] # 해당되는 클러스터 ID들
    if cluster_label == "Single Clsuter":
        selected_cluster_ids = [selected_cluster_ids[0]]
    n_clusters = len(selected_cluster_ids)
    print(f"\n>> [{brain_acronym}] {cluster_label} in region: {n_clusters} clusters found.")
    
    if n_clusters == 0:
        print(f"No {cluster_label.lower()} in '{brain_acronym}' region. Skip.")
        continue
    
    spike_raster, times = compute_raster(spikes, selected_cluster_ids, events, pre_time=2, post_time=4, bin_size=0.001)

    plot_raster(spike_raster, times, events, brain_acronym, cluster_label, save_path)
    plot_psth_2x2(spike_raster, times, left_idx, right_idx, correct_idx, incorrect_idx, brain_acronym, cluster_label, save_path)
    plot_psth_contrast(spike_raster, times, sl.trials, left_idx, right_idx, brain_acronym, cluster_label, save_path)

plt.show(block=False)

close = input("Close all figures? please press any key.")
plt.close('all')
print("Done.")

# -----------------------------------------------------------------------------


# 1. psth (Ntimebins X Ntrials X Nneurons, where Ntimebins encompasses -2s to 4s relative to sensory onset, and each time bin is 1ms, i.e., 6000 time bins
spike_raster_save, times = compute_raster(spikes, selected_mask_good, events, pre_time=2, post_time=4, bin_size=0.001)
spike_raster_save = spike_raster_save.transpose(2, 1, 0) # (Ntimebins X Ntrials X Nneurons) 형태로 transpose
save_file(spike_raster_save, save_path="Z:\ChaeHyeon_Seong", save_title="psth_Ntimebins_Ntrials_Nneurons")

# 2. trial information: stimulus onset time, stimulus contrast, stimulus location (e.g., left/right), stimulus size and orientation, mouse choice (left/right), reaction time etc.
trials = one.load_object(eid, 'trials')
trials_df = trials.to_df()
display(trials_df.head())  # 상위 몇개 미리보기
save_file(trials_df, save_path="Z:\ChaeHyeon_Seong", save_title="trials_info")
# 3. neuron information: hemisphere, area, layer, spike waveform (regular-spiking or fast-spiking)
clusters_df = clusters.to_df()
display(clusters_df.head())  # 상위 몇개 미리보기
save_file(clusters_df, save_path="Z:\ChaeHyeon_Seong", save_title="neuron_info")
# 4. any additional information that might be useful for future analysis