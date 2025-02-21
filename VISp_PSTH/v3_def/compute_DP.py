import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

import os
import sys
sys.path.append('VISp_PSTH/v3_def')
from compute_spike_auc import compute_spike_auc
from compute_mean_roc import compute_mean_roc

def compute_dp(spike_raster, trials_df, mask_contrast, start_idx, end_idx):
    """
    DP(Detection Probability) 계산 및 Mean ROC Curve 생성.
    """
    n_neurons = spike_raster.shape[0]
    dp_values = np.full(n_neurons, np.nan)
    roc_curves = []

    mask_hit = mask_contrast & (trials_df['feedbackType'] == 1)
    mask_miss = mask_contrast & (trials_df['feedbackType'] == -1)

    for i in range(n_neurons):
        fr_hit = np.sum(spike_raster[i, mask_hit, start_idx:end_idx], axis=1) / (end_idx - start_idx)
        fr_miss = np.sum(spike_raster[i, mask_miss, start_idx:end_idx], axis=1) / (end_idx - start_idx)

        # dp_values[i], _, _ = compute_spike_auc(fr_hit, fr_miss)

        # Hit/Miss 개수 맞추기
        n_hit, n_miss = len(fr_hit), len(fr_miss)
        if n_hit == 0 or n_miss == 0:
            continue  # 한쪽 그룹이 없으면 계산 불가

        min_trials = min(n_hit, n_miss)
        fr_hit_balanced = np.random.choice(fr_hit, min_trials, replace=False)
        fr_miss_balanced = np.random.choice(fr_miss, min_trials, replace=False)

        dp_values[i], fpr, tpr = compute_spike_auc(fr_hit_balanced, fr_miss_balanced)
        roc_curves.append((fpr, tpr))

    mean_fpr, mean_tpr = compute_mean_roc(roc_curves)
    return dp_values, (mean_fpr, mean_tpr)