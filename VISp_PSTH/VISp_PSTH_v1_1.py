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
brain_acronym = 'VISp'

# 세션 검색
sessions = one.search(atlas_acronym=brain_acronym, query_type='remote')
print(f'No. of detected sessions: {len(sessions)}\n')
pprint(sessions[:5])  # 첫 5개 세션 출력

# 특정 eid 선택
eid = 'ca4ecb4c-4b60-4723-9b9e-2c54a6290a53'
pids, labels = one.eid2pid(eid)
for pid, name in zip(pids, labels):
    print(f'pid: {pid}, pname: {name}')

# 다른 예시 pid
pid = '92822789-608f-44a6-ad64-fe549402b2df'
# pid = 'c5b9e063-f640-4936-b851-f7602cb6659b'
eid, pname = one.pid2eid(pid)

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

# good 클러스터( label == 1 )
good_cluster_idx = (clusters['label'] == 1)
clusters_good = {key: val[good_cluster_idx] for key, val in clusters.items()}

all_clusters = clusters['label'].shape
good_clusters = clusters_good['label'].shape
print(f'Total no. of clusters: {all_clusters}')
print(f'Number of good clusters: {good_clusters}')

# -----------------------------------------------------------------------------
# 4. Good 클러스터의 뇌 영역(Acronym) 통계
# -----------------------------------------------------------------------------
acronyms = clusters_good['acronym']
unique_acronyms, count = np.unique(acronyms, return_counts=True)
for a, c in zip(unique_acronyms, count):
    print(f'{a}: {c}')

# -----------------------------------------------------------------------------
# 5. Firing rate & 뇌 위치 시각화
# -----------------------------------------------------------------------------
firing_rate = clusters_good['firing_rate']
norm = colors.Normalize(vmin=np.min(firing_rate), vmax=np.max(firing_rate), clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('hot'))
firing_rate_cols = mapper.to_rgba(firing_rate)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# (좌) 채널의 Brain regions
plot_brain_regions(channels['atlas_id'], channel_depths=channels['axial_um'], ax=axs[0])
axs[0].set_title('Brain Regions of Channels')

# (우) 클러스터들의 (amp_median, depth) 산점도 + firing rate 컬러
axs[1].scatter(clusters_good['amp_median'] * 1e6,
               clusters_good['depths'],
               c=firing_rate_cols)
axs[1].set_xlabel('Amplitude (uV)')
axs[1].get_yaxis().set_visible(False)
axs[1].set_title('Firing Rate of Good Clusters')

cbar = fig.colorbar(mapper, ax=axs[1])
cbar.set_label('Firing rate (Hz)')

plt.tight_layout()

# -----------------------------------------------------------------------------
# 6. PSTH (Single Cluster 예시)
# -----------------------------------------------------------------------------
# 6.1 SessionLoader
eid_target, _ = one.pid2eid(pid)  # pid= '92822789-608f-44a6-ad64-fe549402b2df' etc.
sl = SessionLoader(eid=eid_target, one=one)
sl.load_trials()

spikes, clusters, channels = ssl.load_spike_sorting()
clusters = ssl.merge_clusters(spikes, clusters, channels)

# 6.2 선택 기준 (예: 특정 뇌영역 & good_label)
region_str = 'VISp2/3'
good_label = 1

region_mask = (clusters['acronym'] == region_str)
good_mask = (clusters['label'] == good_label)
selected_mask = region_mask & good_mask
selected_cluster_ids = np.where(selected_mask)[0]

print('Trial number:', len(sl.trials))
print(f'Good_cluster number in {region_str}:', len(selected_cluster_ids))

# 자극 이벤트 시간
events = sl.trials['stimOn_times'].values

# 6.3 Spike raster (단일 클러스터)
from brainbox.singlecell import bin_spikes

bin_size = 0.05
pre_time = 1
post_time = 3

single_cluster_id = selected_cluster_ids[3]  # 예시로 n번째 good cluster
single_spikes_idx = (spikes['clusters'] == single_cluster_id)

spike_raster, times = bin_spikes(spikes.times[single_spikes_idx],
                                 events,
                                 pre_time=pre_time,
                                 post_time=post_time,
                                 bin_size=bin_size)
spikes_raster = spike_raster / bin_size

# 6.4 Raster Plot
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(spike_raster,
               extent=[times[0], times[-1], 0, events.size],
               origin='lower', cmap='binary', aspect='auto',
               vmax=20, vmin=0)
ax.axvline(0, c='k', linestyle='--')
ax.set_xlabel('Time from stimulus (s)')
ax.set_ylabel('Trial number')
ax.set_title(f'PSTH of a Single Cluster in {region_str} (Left Hemisphere)')
plt.colorbar(im, ax=ax, label='Spikes (count / bin)')

plt.tight_layout()


# 6.5 PSTH (Left vs Right / Correct vs Incorrect)
left_idx = ~np.isnan(sl.trials['contrastLeft'])
right_idx = ~np.isnan(sl.trials['contrastRight'])

psth_left = np.nanmean(spike_raster[left_idx], axis=0)
psth_right = np.nanmean(spike_raster[right_idx], axis=0)

correct_idx = (sl.trials['feedbackType'] == 1)
incorrect_idx = (sl.trials['feedbackType'] == -1)

psth_correct = np.nanmean(spike_raster[correct_idx], axis=0)
psth_incorrect = np.nanmean(spike_raster[incorrect_idx], axis=0)

# 6.6 시각화: Left vs Right, Correct vs Incorrect
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
fig.suptitle(f'PSTH of a Single Cluster in {region_str} (Left Hemisphere)')

# (좌) Left vs Right
axs[0].plot(times, gaussian_filter1d(psth_left, sigma=1.5), c='crimson', label='Left')
axs[0].plot(times, gaussian_filter1d(psth_right, sigma=1.5), c='darkblue', label='Right')
axs[0].axvline(0, c='k', linestyle='--')
axs[0].set_xlabel('Time from stimulus (s)')
axs[0].set_ylabel('Firing rate (Hz)')
axs[0].set_title('Left vs Right Visual Stimulus')
axs[0].legend()

# (우) Correct vs Incorrect
axs[1].plot(times, gaussian_filter1d(psth_correct, sigma=1.5), c='blueviolet', label='Correct')
axs[1].plot(times, gaussian_filter1d(psth_incorrect, sigma=1.5), c='thistle', label='Incorrect')
axs[1].axvline(0, c='k', linestyle='--')
axs[1].set_xlabel('Time from stimulus (s)')
axs[1].set_title('Correct vs Incorrect Trials')
axs[1].legend()

plt.tight_layout()


# -----------------------------------------------------------------------------
# 7. PSTH (Left/Right + Correct/Incorrect 구분)
# -----------------------------------------------------------------------------
left_correct_idx = left_idx & (sl.trials['feedbackType'] == 1)
left_incorrect_idx = left_idx & (sl.trials['feedbackType'] == -1)
right_correct_idx = right_idx & (sl.trials['feedbackType'] == 1)
right_incorrect_idx = right_idx & (sl.trials['feedbackType'] == -1)

psth_left_correct = np.nanmean(spike_raster[left_correct_idx], axis=0)
psth_left_incorrect = np.nanmean(spike_raster[left_incorrect_idx], axis=0)
psth_right_correct = np.nanmean(spike_raster[right_correct_idx], axis=0)
psth_right_incorrect = np.nanmean(spike_raster[right_incorrect_idx], axis=0)

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
fig.suptitle(f'PSTH of a Single Cluster in {region_str} (Left Hemisphere)')

# (좌) Left correct vs incorrect
axs[0].plot(times, gaussian_filter1d(psth_left_correct, sigma=1.5), c='crimson', label='Left correct')
axs[0].plot(times, gaussian_filter1d(psth_left_incorrect, sigma=1.5), c='rosybrown', label='Left incorrect')
axs[0].axvline(0, c='k', linestyle='--')
axs[0].set_xlabel('Time from stimulus (s)')
axs[0].set_ylabel('Firing rate (Hz)')
axs[0].set_title('Left Stimulus')
axs[0].legend()

# (우) Right correct vs incorrect
axs[1].plot(times, gaussian_filter1d(psth_right_correct, sigma=1.5), c='darkblue', label='Right correct')
axs[1].plot(times, gaussian_filter1d(psth_right_incorrect, sigma=1.5), c='slategray', label='Right incorrect')
axs[1].axvline(0, c='k', linestyle='--')
axs[1].set_xlabel('Time from stimulus (s)')
axs[1].set_title('Right Stimulus')
axs[1].legend()

plt.tight_layout()


# -----------------------------------------------------------------------------
# 8. 모든 뉴런에 대한 PSTH (nClusters, nTrials, nBins)
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

# 8.1 시각화 (imshow)
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle(f'PSTH of All Clusters in {region_str} (Left Hemisphere)')

# (좌) trial별 평균 -> 모든 클러스터 평균
im1 = axs[0].imshow(np.nanmean(spike_raster_all, axis=0),
                    extent=[times[0], times[-1], 0, events.size],
                    origin='lower', cmap='binary', aspect='auto',
                    vmax=20, vmin=0)
axs[0].axvline(0, c='k', linestyle='--')
axs[0].set_xlabel('Time from stimulus (s)')
axs[0].set_ylabel('Trial number')
axs[0].set_title('Mean of All Clusters')
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
# 9. PSTH (All Clusters) - Left/Right & Correct/Incorrect
# -----------------------------------------------------------------------------
left_idx = ~np.isnan(sl.trials['contrastLeft'])
right_idx = ~np.isnan(sl.trials['contrastRight'])

print('Left contrast:', np.unique(sl.trials['contrastLeft'][left_idx]))
print('Right contrast:', np.unique(sl.trials['contrastRight'][right_idx]))

correct_idx = (sl.trials['feedbackType'] == 1)
incorrect_idx = (sl.trials['feedbackType'] == -1)

left_correct_idx = left_idx & correct_idx
left_incorrect_idx = left_idx & incorrect_idx
right_correct_idx = right_idx & correct_idx
right_incorrect_idx = right_idx & incorrect_idx

psth_left = np.nanmean(spike_raster_all[:, left_idx, :], axis=(0, 1))
psth_right = np.nanmean(spike_raster_all[:, right_idx, :], axis=(0, 1))
psth_correct = np.nanmean(spike_raster_all[:, correct_idx, :], axis=(0, 1))
psth_incorrect = np.nanmean(spike_raster_all[:, incorrect_idx, :], axis=(0, 1))

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
fig.suptitle(f'PSTH of All Clusters in {region_str} (Left Hemisphere)')

# (좌) Left vs Right
axs[0].plot(times, gaussian_filter1d(psth_left, sigma=1.5), c='crimson', label='Left')
axs[0].plot(times, gaussian_filter1d(psth_right, sigma=1.5), c='darkblue', label='Right')
axs[0].axvline(0, c='k', linestyle='--')
axs[0].set_xlabel('Time from stimulus (s)')
axs[0].set_ylabel('Firing rate (Hz)')
axs[0].set_title('Left vs Right Visual Stimulus')
axs[0].legend()

# (우) Correct vs Incorrect
axs[1].plot(times, gaussian_filter1d(psth_correct, sigma=1.5), c='blueviolet', label='Correct')
axs[1].plot(times, gaussian_filter1d(psth_incorrect, sigma=1.5), c='thistle', label='Incorrect')
axs[1].axvline(0, c='k', linestyle='--')
axs[1].set_xlabel('Time from stimulus (s)')
axs[1].set_title('Correct vs Incorrect Trials')
axs[1].legend()

plt.tight_layout()


# -----------------------------------------------------------------------------
# 9.1 Left/Right + Correct/Incorrect 분할
# -----------------------------------------------------------------------------
psth_left_correct = np.nanmean(spike_raster_all[:, left_correct_idx, :], axis=(0, 1))
psth_left_incorrect = np.nanmean(spike_raster_all[:, left_incorrect_idx, :], axis=(0, 1))
psth_right_correct = np.nanmean(spike_raster_all[:, right_correct_idx, :], axis=(0, 1))
psth_right_incorrect = np.nanmean(spike_raster_all[:, right_incorrect_idx, :], axis=(0, 1))

fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
fig.suptitle(f'PSTH of All Clusters in {region_str} (Left Hemisphere)')

# Left correct vs incorrect
axs[0].plot(times, gaussian_filter1d(psth_left_correct, sigma=1.5),
            c='crimson', label='Left correct')
axs[0].plot(times, gaussian_filter1d(psth_left_incorrect, sigma=1.5),
            c='rosybrown', label='Left incorrect')
axs[0].axvline(0, c='k', linestyle='--')
axs[0].set_xlabel('Time from stimulus onset (s)')
axs[0].set_ylabel('Firing rate (Hz)')
axs[0].set_title('Left Stimulus')
axs[0].legend()

# Right correct vs incorrect
axs[1].plot(times, gaussian_filter1d(psth_right_correct, sigma=1.5),
            c='darkblue', label='Right correct')
axs[1].plot(times, gaussian_filter1d(psth_right_incorrect, sigma=1.5),
            c='slategray', label='Right incorrect')
axs[1].axvline(0, c='k', linestyle='--')
axs[1].set_xlabel('Time from stimulus onset (s)')
axs[1].set_title('Right Stimulus')
axs[1].legend()

plt.tight_layout()


# -----------------------------------------------------------------------------
# 10. Contrast별 PSTH (All Clusters)
# -----------------------------------------------------------------------------
contrasts_left = np.unique(sl.trials['contrastLeft'][left_idx])     # [0.0, 0.0625, 0.125, 0.25, 1.0]
contrasts_right = np.unique(sl.trials['contrastRight'][right_idx])  # 대개 동일

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
fig.suptitle(f'PSTH of All Clusters in {region_str} (Left Hemisphere)')

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
