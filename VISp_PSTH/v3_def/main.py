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
from brainbox.task.trials import find_trial_ids, get_event_aligned_raster, get_psth, filter_by_trial, filter_correct_incorrect_left_right, filter_correct_incorrect, filter_left_right, filter_trials
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

from plot_psth_2x2 import plot_psth_2x2



# -----------------------------------------------------------------------------
# 2. 데이터 로드 및 전처리
# -----------------------------------------------------------------------------
brain_acronym = 'VISp'
one = ONE()

eid = 'ca4ecb4c-4b60-4723-9b9e-2c54a6290a53'
pid = one.eid2pid(eid)[0][0]

sl = SessionLoader(eid=eid, one=one)
sl.load_trials()
trials_df = sl.trials
# column 제목들만 추출
print('Keys of trials:', list(trials_df.keys()))
display(trials_df.head())

# trial_id, dividers = find_trial_ids(trials_df, side='all', choice='right', order='trial num', sort='idx', contrast=(1, 0.5, 0.25, 0.125, 0.0625, 0), event=None)

ssl = SpikeSortingLoader(one=one, pid=pid, atlas=AllenAtlas())
spikes, clusters, channels = ssl.load_spike_sorting(good_units=True)
clusters = ssl.merge_clusters(spikes, clusters, channels)

pre_time=2.0
post_time=4.0
bin_size=0.001
events=trials_df['stimOn_times'].values
times = np.arange(-pre_time, post_time, bin_size)
event_raster, times = get_event_aligned_raster(times, events, tbin=0.001, values=spikes, epoch=[-2.0, 4.0], bin=True)
# event_raster, psth = filter_correct_incorrect_left_right(trials_df, event_raster, events, contrast=(1, 0.5, 0.25, 0.125, 0.0625, 0), order='trial num')
# event_raster = filter_by_trial(event_raster, trial_id)
mean, err = get_psth(event_raster, trial_ids=None)

# -----------------------------------------------------------------------------
# 3. PSTH Plot
# -----------------------------------------------------------------------------

plot_psth_2x2(mean, err, times, contrast=(1, 0.5, 0.25, 0.125, 0.0625, 0))
plt.show()

# 아무 키나 누르면 피규어 창이 닫힙니다.
input("Press any key to close the figure...\n")
plt.close()

