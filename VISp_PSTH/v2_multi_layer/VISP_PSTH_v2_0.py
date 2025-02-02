import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from scipy.ndimage import gaussian_filter1d
from IPython.display import display

# brainbox / iblatlas / ONE 관련
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from brainbox.singlecell import bin_spikes
from brainbox.ephys_plots import plot_brain_regions
from iblatlas.atlas import AllenAtlas
from one.api import ONE

from matplotlib import cm, colors

# -----------------------------------------------------------------------------
# 1. ONE 초기화 및 세션 검색
# -----------------------------------------------------------------------------
one = ONE()
brain_acronym = 'VISp'  # 이 문자열이 들어간 모든 VISp 영역 대상

# VISp 관련 세션 검색
sessions = one.search(atlas_acronym=brain_acronym, query_type='remote')
print(f'\nNo. of detected sessions in {brain_acronym}: {len(sessions)}\n')
pprint(sessions[0:5])

# 특정 eid 선택 (예시)
eid = 'ca4ecb4c-4b60-4723-9b9e-2c54a6290a53'
pids, labels = one.eid2pid(eid)
for pid, name in zip(pids, labels):
    print(f'pid: {pid}, pname: {name}')

# 관련 pid (예시)
pid = '92822789-608f-44a6-ad64-fe549402b2df'
eid, pname = one.pid2eid(pid)
print(f'eid: {eid}, pname: {pname}')

# -----------------------------------------------------------------------------
# 2. Trial 정보 로드 및 표시
# -----------------------------------------------------------------------------
trials = one.load_object(eid, 'trials')
print('Keys of trials:', list(trials.keys()))

trials_df = trials.to_df()
display(trials_df)  # IPython.display 사용

# -----------------------------------------------------------------------------
# 3. Spike Sorting 데이터 로드 / good 클러스터 추출
# -----------------------------------------------------------------------------
ba = AllenAtlas()
ssl = SpikeSortingLoader(one=one, pid=pid, atlas=ba)
spikes, clusters, channels = ssl.load_spike_sorting()

# 클러스터 병합
clusters = ssl.merge_clusters(spikes, clusters, channels)

# good 클러스터( label == 1 ) 필터
good_cluster_idx = (clusters['label'] == 1)
clusters_good = {key: val[good_cluster_idx] for key, val in clusters.items()}

all_clusters = clusters['label'].shape
good_clusters = clusters_good['label'].shape

# -----------------------------------------------------------------------------
# 4. Clusters / Good Clusters의 뇌 영역(Acronym) 통계
# -----------------------------------------------------------------------------
print(f'\nNo. of all clusters in the session: {all_clusters}')
acronyms = clusters['acronym']
unique_acronyms, count = np.unique(acronyms, return_counts=True)
print('\nNo. of clusters in each region:')
num_clusters = 0
for a, c in zip(unique_acronyms, count):
    print(f'{a}: {c}')
    if brain_acronym in a:  # brain_acronym='VISp'가 포함되어 있으면
        num_clusters += c
print(f'\nNo. of clusters in {brain_acronym}: {num_clusters}')


print(f'\nNo. of good clusters in the session: {good_clusters}')
acronyms_good = clusters_good['acronym']
unique_acronyms_good, count_good = np.unique(acronyms_good, return_counts=True)
print('\nNo. of good clusters in each region:')
num_good_clusters = 0
for a, c in zip(unique_acronyms_good, count_good):
    print(f'{a}: {c}')
    if brain_acronym in a:
        num_good_clusters += c

print(f'\nNo. of good clusters in {brain_acronym}: {num_good_clusters}')

# -----------------------------------------------------------------------------
# 5. SessionLoader (Trial, event 정보 로드)
# -----------------------------------------------------------------------------
eid_target, _ = one.pid2eid(pid)  # 예시 pid
sl = SessionLoader(eid=eid_target, one=one)
sl.load_trials()

# 다시 Spike Sorting 로드(혹은 reuse)
spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels)

# -----------------------------------------------------------------------------
# 6. 특정 뇌영역(VISp*), good_label 조건에 맞는 clusters 선택
# -----------------------------------------------------------------------------
# region_mask: clusters['acronym']에 'VISp' 문자열이 포함된 경우
region_mask = np.array(['VISp' in acr for acr in clusters['acronym']])
good_label = 1
good_mask = (clusters['label'] == good_label)

# 최종 선택 마스크
selected_mask = region_mask & good_mask
selected_cluster_ids = np.where(selected_mask)[0]

print('Trial number:', len(sl.trials))
print(f'Selected_cluster number in all VISp-subregions:', len(selected_cluster_ids))

# 자극 이벤트 시간
events = sl.trials['stimOn_times'].values

# 파라미터 설정
bin_size = 0.05
pre_time = 1
post_time = 3

# -----------------------------------------------------------------------------
# 6.1 Spike raster & PSTH (단일 클러스터 예시)
# -----------------------------------------------------------------------------
from brainbox.singlecell import bin_spikes

# 일단 하나 선택(예: 3번째)
if len(selected_cluster_ids) > 3:
    single_cluster_id = selected_cluster_ids[3]
else:
    single_cluster_id = selected_cluster_ids[0]

single_spikes_idx = (spikes['clusters'] == single_cluster_id)

spike_raster, times = bin_spikes(spikes.times[single_spikes_idx],
                                 events,
                                 pre_time=pre_time,
                                 post_time=post_time,
                                 bin_size=bin_size)
spikes_raster = spike_raster / bin_size  # firing rate

# Raster Plot
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(spike_raster,
               extent=[times[0], times[-1], 0, events.size],
               origin='lower', cmap='binary', aspect='auto',
               vmax=20, vmin=0)
ax.axvline(0, c='k', linestyle='--')
ax.set_xlabel('Time from stimulus (s)')
ax.set_ylabel('Trial number')
ax.set_title(f'Spike Raster of a Single Cluster (All VISp regions)')
plt.colorbar(im, ax=ax, label='Spikes (count / bin)')
plt.tight_layout()

# PSTH
left_idx = ~np.isnan(sl.trials['contrastLeft'])
right_idx = ~np.isnan(sl.trials['contrastRight'])
correct_idx = (sl.trials['feedbackType'] == 1)
incorrect_idx = (sl.trials['feedbackType'] == -1)

psth_left = np.nanmean(spike_raster[left_idx], axis=0)
psth_right = np.nanmean(spike_raster[right_idx], axis=0)
psth_correct = np.nanmean(spike_raster[correct_idx], axis=0)
psth_incorrect = np.nanmean(spike_raster[incorrect_idx], axis=0)

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
fig.suptitle('PSTH of a Single Cluster (All VISp regions)')

# Left vs Right
axs[0].plot(times, gaussian_filter1d(psth_left, sigma=1.5), c='crimson', label='Left')
axs[0].plot(times, gaussian_filter1d(psth_right, sigma=1.5), c='darkblue', label='Right')
axs[0].axvline(0, c='k', linestyle='--')
axs[0].set_xlabel('Time from stimulus (s)')
axs[0].set_ylabel('Firing rate (Hz)')
axs[0].set_title('Left vs Right Visual Stimulus')
axs[0].legend()

# Correct vs Incorrect
axs[1].plot(times, gaussian_filter1d(psth_correct, sigma=1.5), c='blueviolet', label='Correct')
axs[1].plot(times, gaussian_filter1d(psth_incorrect, sigma=1.5), c='thistle', label='Incorrect')
axs[1].axvline(0, c='k', linestyle='--')
axs[1].set_xlabel('Time from stimulus (s)')
axs[1].set_title('Correct vs Incorrect Trials')
axs[1].legend()

plt.tight_layout()

# -----------------------------------------------------------------------------
# 7. Spike raster (All selected clusters = All VISp-subregions)
# -----------------------------------------------------------------------------
spike_raster_all = []
for cid in selected_cluster_ids:
    spikes_idx = (spikes['clusters'] == cid)
    spike_times_cluster = spikes['times'][spikes_idx]

    spike_raster_cid, _ = bin_spikes(spike_times_cluster,
                                     events,
                                     pre_time=pre_time,
                                     post_time=post_time,
                                     bin_size=bin_size)
    spike_raster_hz = spike_raster_cid / bin_size
    spike_raster_all.append(spike_raster_hz)

spike_raster_all = np.stack(spike_raster_all, axis=0)  # shape = (nClusters, nTrials, nBins)
print('spike_raster_all.shape:', spike_raster_all.shape)

# (뉴런, trial) 평균 => PSTH
psth_all = np.nanmean(spike_raster_all, axis=(0, 1))

# 시각화 (imshow) - (좌) trial별 평균, (우) 클러스터별 평균
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Spike Rasters of All VISp Clusters')

# (좌) trial별 평균 -> 모든 클러스터 평균
im1 = axs[0].imshow(np.nanmean(spike_raster_all, axis=0),
                    extent=[times[0], times[-1], 0, events.size],
                    origin='lower', cmap='binary', aspect='auto',
                    vmax=20, vmin=0)
axs[0].axvline(0, c='k', linestyle='--')
axs[0].set_xlabel('Time from stimulus (s)')
axs[0].set_ylabel('Trial number')
axs[0].set_title('Mean of All VISp Clusters')
plt.colorbar(im1, ax=axs[0])

# (우) 클러스터별 평균 -> 모든 trial 평균
im2 = axs[1].imshow(np.nanmean(spike_raster_all, axis=1),
                    extent=[times[0], times[-1], 0, len(selected_cluster_ids)],
                    origin='lower', cmap='binary', aspect='auto',
                    vmax=20, vmin=0)
axs[1].axvline(0, c='k', linestyle='--')
axs[1].set_xlabel('Time from stimulus (s)')
axs[1].set_ylabel('Cluster index')
axs[1].set_title('Mean of All Trials')
plt.colorbar(im2, ax=axs[1])

plt.tight_layout()

# -----------------------------------------------------------------------------
# 8. PSTH (All VISp Clusters) - Left/Right & Correct/Incorrect
# -----------------------------------------------------------------------------
left_idx = ~np.isnan(sl.trials['contrastLeft'])
right_idx = ~np.isnan(sl.trials['contrastRight'])
correct_idx = (sl.trials['feedbackType'] == 1)
incorrect_idx = (sl.trials['feedbackType'] == -1)

psth_left = np.nanmean(spike_raster_all[:, left_idx, :], axis=(0, 1))
psth_right = np.nanmean(spike_raster_all[:, right_idx, :], axis=(0, 1))
psth_correct = np.nanmean(spike_raster_all[:, correct_idx, :], axis=(0, 1))
psth_incorrect = np.nanmean(spike_raster_all[:, incorrect_idx, :], axis=(0, 1))

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
fig.suptitle('PSTH of All VISp Clusters')

# Left vs Right
axs[0].plot(times, gaussian_filter1d(psth_left, sigma=1.5), c='crimson', label='Left')
axs[0].plot(times, gaussian_filter1d(psth_right, sigma=1.5), c='darkblue', label='Right')
axs[0].axvline(0, c='k', linestyle='--')
axs[0].set_xlabel('Time from stimulus (s)')
axs[0].set_ylabel('Firing rate (Hz)')
axs[0].set_title('Left vs Right Visual Stimulus')
axs[0].legend()

# Correct vs Incorrect
axs[1].plot(times, gaussian_filter1d(psth_correct, sigma=1.5), c='blueviolet', label='Correct')
axs[1].plot(times, gaussian_filter1d(psth_incorrect, sigma=1.5), c='thistle', label='Incorrect')
axs[1].axvline(0, c='k', linestyle='--')
axs[1].set_xlabel('Time from stimulus (s)')
axs[1].set_title('Correct vs Incorrect Trials')
axs[1].legend()

plt.tight_layout()

# -----------------------------------------------------------------------------
# 9. PSTH (All VISp Clusters) - Left/Right + Correct/Incorrect 분할
# -----------------------------------------------------------------------------
left_correct_idx = left_idx & correct_idx
left_incorrect_idx = left_idx & incorrect_idx
right_correct_idx = right_idx & correct_idx
right_incorrect_idx = right_idx & incorrect_idx

psth_left_correct = np.nanmean(spike_raster_all[:, left_correct_idx, :], axis=(0, 1))
psth_left_incorrect = np.nanmean(spike_raster_all[:, left_incorrect_idx, :], axis=(0, 1))
psth_right_correct = np.nanmean(spike_raster_all[:, right_correct_idx, :], axis=(0, 1))
psth_right_incorrect = np.nanmean(spike_raster_all[:, right_incorrect_idx, :], axis=(0, 1))

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
fig.suptitle('PSTH of All VISp Clusters')

# Left correct vs incorrect
axs[0].plot(times, gaussian_filter1d(psth_left_correct, sigma=1.5), c='crimson', label='Left correct')
axs[0].plot(times, gaussian_filter1d(psth_left_incorrect, sigma=1.5), c='rosybrown', label='Left incorrect')
axs[0].axvline(0, c='k', linestyle='--')
axs[0].set_xlabel('Time from stimulus (s)')
axs[0].set_ylabel('Firing rate (Hz)')
axs[0].set_title('Left Stimulus')
axs[0].legend()

# Right correct vs incorrect
axs[1].plot(times, gaussian_filter1d(psth_right_correct, sigma=1.5), c='darkblue', label='Right correct')
axs[1].plot(times, gaussian_filter1d(psth_right_incorrect, sigma=1.5), c='slategray', label='Right incorrect')
axs[1].axvline(0, c='k', linestyle='--')
axs[1].set_xlabel('Time from stimulus (s)')
axs[1].set_title('Right Stimulus')
axs[1].legend()

plt.tight_layout()

# -----------------------------------------------------------------------------
# 10. Contrast별 PSTH (All VISp Clusters)
# -----------------------------------------------------------------------------
contrasts_left = np.unique(sl.trials['contrastLeft'][left_idx])     
contrasts_right = np.unique(sl.trials['contrastRight'][right_idx])  

psth_list_left = []
for c in contrasts_left:
    c_mask = left_idx & (sl.trials['contrastLeft'] == c)
    psth_tmp = np.nanmean(spike_raster_all[:, c_mask, :], axis=(0, 1))
    psth_smooth = gaussian_filter1d(psth_tmp, sigma=1.5)
    psth_list_left.append(psth_smooth)

psth_list_right = []
for c in contrasts_right:
    c_mask = right_idx & (sl.trials['contrastRight'] == c)
    psth_tmp = np.nanmean(spike_raster_all[:, c_mask, :], axis=(0, 1))
    psth_smooth = gaussian_filter1d(psth_tmp, sigma=1.5)
    psth_list_right.append(psth_smooth)

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
fig.suptitle('Contrast-based PSTH (All VISp Clusters)')

# Left stimulus
for i, c_val in enumerate(contrasts_left):
    color_val = i / (len(contrasts_left) - 1) if len(contrasts_left) > 1 else 1.0
    color = plt.cm.Reds(color_val)
    axs[0].plot(times, psth_list_left[i], label=f'{c_val*100:.1f}%', c=color)

axs[0].axvline(0, c='k', linestyle='--')
axs[0].set_xlabel('Time from stimulus (s)')
axs[0].set_ylabel('Firing rate (Hz)')
axs[0].set_title('Left Stimulus')
axs[0].legend(title='Contrast')

# Right stimulus
for i, c_val in enumerate(contrasts_right):
    color_val = i / (len(contrasts_right) - 1) if len(contrasts_right) > 1 else 1.0
    color = plt.cm.Blues(color_val)
    axs[1].plot(times, psth_list_right[i], label=f'{c_val*100:.1f}%', c=color)

axs[1].axvline(0, c='k', linestyle='--')
axs[1].set_xlabel('Time from stimulus (s)')
axs[1].set_title('Right Stimulus')
axs[1].legend(title='Contrast')

plt.tight_layout()
plt.show()
