import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from pprint import pprint
from scipy.ndimage import gaussian_filter1d
from IPython.display import display

# brainbox / iblatlas / ONE 관련
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from brainbox.singlecell import bin_spikes
from brainbox.ephys_plots import plot_brain_regions
from iblatlas.atlas import AllenAtlas
from one.api import ONE

import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('./v3_def')
from print_cluster_info import print_cluster_info
from get_trial_masks import get_trial_masks
from compute_raster import compute_raster
from plot_raster import plot_raster
from plot_psth_2x2 import plot_psth_2x2
from plot_psth_contrast import plot_psth_contrast
from sub_func import save_file

save_path = r"C:\Users\miasc\SCH\shinlab\ChaeHyeon_Seong\dataset_v2"

one = ONE()

# 원하는 뇌 영역 (VISp)
brain_acronym = 'VISp'
sessions = one.search(atlas_acronym=brain_acronym, query_type='remote')
sessions = list(sessions)
for i in [44, 45, 68, 68, 68]:
    sessions.pop(i)
print(f"Found {len(sessions)} sessions for region: {brain_acronym}")

# ---------------------------------------------------------------
# 세션 반복
# ---------------------------------------------------------------
i = 1
for eid in sessions:
    # 세션 10가 되면 종료
    if i == 10:
        break
    print(f"\n=== [Session:{i} {eid}] ===")
    i += 1

    session_folder = os.path.join(save_path, eid)
    os.makedirs(session_folder, exist_ok=True)
    
    session_path = str(one.eid2path(eid))
    with open(os.path.join(session_folder, "session_path.txt"), "w", encoding="utf-8") as f:
        f.write(session_path)

    # -----------------------------------------------------------------
    # 1) Trials 정보 로드 & 저장
    # -----------------------------------------------------------------
    sl = SessionLoader(eid=eid, one=one)
    sl.load_trials()
    trials_df = sl.trials

    # --- basic task와 full task 구분 ---
    if 'probabilityLeft' in trials_df.columns:
        unique_prob = np.unique(trials_df['probabilityLeft'].dropna())
        if len(unique_prob) == 1 and np.isclose(unique_prob[0], 0.5):
            task_type = 'basic'
        else:
            task_type = 'full'
    else:
        task_type = 'unknown'
    print(f"Session {eid} is identified as a {task_type} task.")
    with open(os.path.join(session_folder, "task_type.txt"), "w", encoding="utf-8") as f:
        f.write(task_type)

    trials_df_selected = trials_df[['stimOn_times', 'contrastLeft', 'contrastRight',
                                    'choice', 'feedbackType', 'response_times']].copy()
    trials_df_selected['task_type'] = task_type

    save_file(trials_df_selected,
              save_path=session_folder,
              save_title="trials_info")

    # -----------------------------------------------------------------
    # 2) Spike/Cluster 정보: Probe 별로 반복
    # -----------------------------------------------------------------
    pids, labels = one.eid2pid(eid)
    if len(pids) == 0:
        print(f" - No probe data found in session {eid}, skip.")
        continue

    events = sl.trials['stimOn_times'].values

    for pid, label in zip(pids, labels):
        print(f"   -> Probe: {pid} ({label})")
        probe_folder = os.path.join(session_folder, label)
        os.makedirs(probe_folder, exist_ok=True)

        clusters_from_obj = one.load_object(eid, 'clusters', collection=f'alf/{label}/pykilosort')
        peak2trough = clusters_from_obj.peakToTrough

        ssl = SpikeSortingLoader(one=one, pid=pid, atlas=AllenAtlas())
        spikes, clusters, channels = ssl.load_spike_sorting()
        clusters = ssl.merge_clusters(spikes, clusters, channels)

        # -----------------------------
        # 2-1) 뉴런 정보와 spike waveform 분류 및 전체 recording FR 계산
        # -----------------------------
        clusters_df = clusters.to_df()
        clusters_df['peakToTrough'] = peak2trough
        threshold = 0.5  # 임계값 (ms)
        clusters_df['spikeType'] = clusters_df['peakToTrough'].abs().apply(
            lambda x: 'fast-spiking' if x < threshold else 'regular-spiking'
        )

        # 전체 recording에서 각 뉴런의 평균 firing rate 계산
        # spikes.times는 session 전체의 spike 시간 (초 단위)
        # 각 unit에 대해 spike 개수를 세고, 전체 recording 기간(마지막 - 첫 spike 시간)으로 나눔
        recording_duration = spikes.times[-1] - spikes.times[0]
        unique_units = np.unique(spikes.clusters)
        mean_fr_dict = {unit: np.sum(spikes.clusters == unit) / recording_duration 
                        for unit in unique_units}
        # clusters_df의 인덱스 혹은 'id' 컬럼을 사용하여 매핑 (여기서는 index를 사용)
        clusters_df['meanFR_session'] = clusters_df.index.map(lambda x: mean_fr_dict.get(x, np.nan))

        save_file(clusters_df, save_path=probe_folder, save_title="neuron_info")
        # 전체 session FR를 별도로 저장 (Method 1용)
        np.save(os.path.join(probe_folder, "mean_fr_session.npy"), 
                clusters_df['meanFR_session'].values)

        # -------------------------------------------------------------
        # 2-2) PSTH 계산 (trial window 기반은 그대로 두되, Method 2, 3 계산은 그대로 수행)
        # -------------------------------------------------------------
        pre_time = 2.0
        post_time = 4.0
        bin_size = 0.001  # 1ms
        # psth의 시간 배열 (-2 ~ +4초)
        # (참고: 이 PSTH는 trial-locked window로 계산됨)
        # times = np.arange(-pre_time, post_time, bin_size)  <- 이미 compute_raster 내부에서 사용됨

        spike_raster, time_bins = compute_raster(spikes, 
                                                  np.where((clusters_df.index.values)[clusters_df.index.isin(unique_units)])[0],
                                                  events,
                                                  pre_time=pre_time,
                                                  post_time=post_time,
                                                  bin_size=bin_size)
        save_file(spike_raster, save_path=probe_folder, save_title="psth")
        save_file(time_bins, save_path=probe_folder, save_title="time_bins")

print("\n\nAll done!")
