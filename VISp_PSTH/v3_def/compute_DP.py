import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

def compute_DP(spike_raster, times, trials_df, response_window=(0, 0.2)):
    """
    각 뉴런별 DP (Detect Probability)를 계산합니다.
    spike_raster: shape = [nNeurons, nTrials, nBins]
    times: 각 bin의 시간 (예: -2초 ~ 4초)
    trials_df: trials 정보가 들어있는 DataFrame (feedbackType 컬럼 필요; hit: 1, miss: -1)
    response_window: 자극 onset 후 반응을 측정할 시간 창 (초)
    """
    # 반응창 내 시간 인덱스
    idx_window = (times >= response_window[0]) & (times < response_window[1])
    
    # hit, miss trial 마스크 (hit: feedbackType==1, miss: feedbackType==-1)
    feedback = trials_df['feedbackType'].values  # 예: 1 또는 -1
    hit_mask = (feedback == 1)
    miss_mask = (feedback == -1)
    valid_mask = hit_mask | miss_mask  # hit 또는 miss인 trial만 사용
    
    nNeurons = spike_raster.shape[0]
    dp_values = np.zeros(nNeurons)
    
    for neuron in range(nNeurons):
        # 각 trial에서 response window 내 spike count 계산
        counts = np.sum(spike_raster[neuron][:, idx_window], axis=1)
        counts_valid = counts[valid_mask]
        labels = feedback[valid_mask]
        # roc_auc_score는 레이블을 0과 1로 사용하므로 hit:1, miss:0
        labels_binary = (labels == 1).astype(int)
        
        # 만약 hit와 miss trial 모두 충분하지 않으면 NaN 처리
        if len(np.unique(labels_binary)) < 2:
            dp = np.nan
        else:
            dp = roc_auc_score(labels_binary, counts_valid)
        dp_values[neuron] = dp

    return dp_values
