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
# 세션 검색 (atlas_acronym=brain_acronym)
# 필요하면 query_type='remote' 제거 또는 추가 등으로 조정
sessions = one.search(atlas_acronym=brain_acronym, query_type='remote')
# list로 변환
sessions = list(sessions)
# 44번째 세션 제거
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

    # 세션별 폴더 생성
    session_folder = os.path.join(save_path, eid)
    os.makedirs(session_folder, exist_ok=True)
    
    # 세션 경로 저장
    session_path = str(one.eid2path(eid))
    with open(os.path.join(session_folder, "session_path.txt"), "w", encoding="utf-8") as f:
        f.write(session_path)

    # -----------------------------------------------------------------
    # 1) Trials 정보 로드 & 저장
    # -----------------------------------------------------------------
    sl = SessionLoader(eid=eid, one=one)
    sl.load_trials()
    trials_df = sl.trials

    # --- 여기서 basic task와 full task 구분 ---
    if 'probabilityLeft' in trials_df.columns:
        unique_prob = np.unique(trials_df['probabilityLeft'].dropna())
        # 기본 태스크: probabilityLeft가 오직 0.5만을 가진 경우
        if len(unique_prob) == 1 and np.isclose(unique_prob[0], 0.5):
            task_type = 'basic'
        else:
            task_type = 'full'
    else:
        task_type = 'unknown'
    print(f"Session {eid} is identified as a {task_type} task.")
    # task_type 정보를 파일로 저장
    with open(os.path.join(session_folder, "task_type.txt"), "w", encoding="utf-8") as f:
        f.write(task_type)
    # --------------------------------------------------

    # 필요한 열만 선택하여 trials_df_selected 생성
    trials_df_selected = trials_df[['stimOn_times', 'contrastLeft', 'contrastRight',
                                    'choice', 'feedbackType', 'response_times']].copy()
    # 기본 태스크와 전체 태스크 구분 정보도 추가할 수 있다면
    trials_df_selected['task_type'] = task_type

    save_file(trials_df_selected,
                save_path=session_folder,
                save_title="trials_info")

    # -----------------------------------------------------------------
    # 2) Spike/Cluster 정보: Probe 별로 반복
    # -----------------------------------------------------------------
    # 한 세션에 여러 probe가 있을 수 있으므로, pid 목록을 가져옴
    pids, labels = one.eid2pid(eid)
    if len(pids) == 0:
        print(f" - No probe data found in session {eid}, skip.")
        continue

    # 이벤트(자극 on) 추출
    events = sl.trials['stimOn_times'].values

    # 세션 안에 있는 각 probe 마다 반복
    for pid, label in zip(pids, labels):
        print(f"   -> Probe: {pid} ({label})")
        # probe마다 폴더 생성
        probe_folder = os.path.join(session_folder, label)
        os.makedirs(probe_folder, exist_ok=True)

        # ONE을 통해 clusters 정보를 로드 (pykilosort collection)
        clusters_from_obj = one.load_object(eid, 'clusters', collection=f'alf/{label}/pykilosort')
        peak2trough = clusters_from_obj.peakToTrough

        # Spike Sorting 로드
        ssl = SpikeSortingLoader(one=one, pid=pid, atlas=AllenAtlas())
        spikes, clusters, channels = ssl.load_spike_sorting()
        clusters = ssl.merge_clusters(spikes, clusters, channels)

        # -------------------------------------------------------------
        # 2-1) 뉴런 정보 & spike waveform (fast/regular) 구분
        # -------------------------------------------------------------
        clusters_df = clusters.to_df()
        clusters_df['peakToTrough'] = peak2trough

        threshold = 0.5  # 예시 임계값 (ms)
        clusters_df['spikeType'] = clusters_df['peakToTrough'].abs().apply(
            lambda x: 'fast-spiking' if x < threshold else 'regular-spiking'
        )

        save_file(clusters_df, save_path=probe_folder, save_title="neuron_info")

        # -------------------------------------------------------------
        # 2-2) PSTH 계산 (Ntimebins x Ntrials x Nneurons)
        # -------------------------------------------------------------
        pre_time = 2.0
        post_time = 4.0
        bin_size = 0.001  # 1ms

        # 예: VISp 영역인 클러스터만 선택 (문자열 검색)
        region_mask = np.array([brain_acronym in acr for acr in clusters['acronym']])
        # Good cluster mask
        good_cluster_mask = (clusters['label'] == 1)
        selected_mask = np.where(region_mask & good_cluster_mask)[0]

        spike_raster, time_bins = compute_raster(spikes, selected_mask, events,
                                                  pre_time=pre_time,
                                                  post_time=post_time,
                                                  bin_size=bin_size)

        save_file(spike_raster, save_path=probe_folder, save_title="psth")
        save_file(time_bins, save_path=probe_folder, save_title="time_bins")

print("\n\nAll done!")
