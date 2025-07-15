import numpy as np
from brainbox.singlecell import bin_spikes

from sub_func import save_file, _shorten_title


def compute_raster(spikes, selected_mask, events,
                   pre_time=1.0, post_time=3.0, bin_size=0.05):
    """
    여러 클러스터(또는 단일 클러스터)에 대해 spike raster를 구한 뒤
    shape = [nUnits, nTrials, nBins] 형태로 반환.
      - 단일 클러스터 ID만 주면 nUnits=1
      - 복수 클러스터 ID를 주면 nUnits=(number of clusters)
    times: bin의 시간축 (pre_time ~ post_time)
    """
    # cluster_ids가 스칼라(단일 int, float, numpy.int64 등)이면 리스트로 변환
    if np.isscalar(selected_mask):
        selected_mask = [selected_mask]
    
    raster_list = []
    for cid in selected_mask:
        mask = (spikes['clusters'] == cid)
        spike_raster_c, _ = bin_spikes(spikes['times'][mask],
                                       events,
                                       pre_time=pre_time,
                                       post_time=post_time,
                                       bin_size=bin_size)
        # count/bin -> Hz
        raster_list.append(spike_raster_c / bin_size)

    # [nUnits, nTrials, nBins] 형태
    if len(raster_list) == 0:
        # 선택된 클러스터가 없는 경우, 빈 배열 반환 (shape: [0, nTrials, nBins])
        nTrials = len(events)      # events 배열에 포함된 trial 수
        nBins = len(np.arange(-pre_time, post_time, bin_size))     # 시간 bin의 개수
        spike_raster = np.empty((0, nTrials, nBins))
    else:
        # [nUnits, nTrials, nBins] 형태로 stack
        spike_raster = np.stack(raster_list, axis=0)

    # time array
    times = np.arange(-pre_time, post_time, bin_size)
    
    return spike_raster, times

