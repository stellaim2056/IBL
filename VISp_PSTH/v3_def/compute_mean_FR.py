import numpy as np

def compute_mean_FR(spike_raster, times, window=(0, 0.2)):
    """
    각 뉴런별 mean firing rate (Hz)를 계산합니다.
    spike_raster: shape = [nNeurons, nTrials, nBins]
    times: 각 bin의 시간 값
    window: firing rate를 계산할 시간 창 (초)
    """
    idx_window = (times >= window[0]) & (times < window[1])
    # 각 trial의 spike count 합계 후, firing rate = count / window duration, trial 평균
    nNeurons = spike_raster.shape[0]
    mean_FR = np.zeros(nNeurons)
    for neuron in range(nNeurons):
        # trial별 firing rate
        fr_trials = np.sum(spike_raster[neuron][:, idx_window], axis=1) / (window[1]-window[0])
        mean_FR[neuron] = np.mean(fr_trials)
    return mean_FR