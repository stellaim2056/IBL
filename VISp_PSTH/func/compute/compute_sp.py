import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

import os
import sys
sys.path.append('VISp_PSTH/v3_def')
from compute_spike_auc import compute_spike_auc
from compute_mean_roc import compute_mean_roc

def compute_sp(spike_raster, trials_df, start_idx, end_idx):
    """
    SP(Stimulus Probability) 계산 및 Mean ROC Curve 생성.
    """
    mask_0 = (trials_df['contrastRight'] == 0)
    mask_100 = (trials_df['contrastRight'] == 1)

    bin_size = 0.001
    n_neurons = spike_raster.shape[0]
    sp_values = np.full(n_neurons, np.nan)
    roc_curves = []

    for i in range(n_neurons):
        fr_0 = np.sum(spike_raster[i, mask_0, start_idx:end_idx], axis=1) / (end_idx - start_idx) / bin_size # fr_0.shape = (n_trials,)
        fr_100 = np.sum(spike_raster[i, mask_100, start_idx:end_idx], axis=1) / (end_idx - start_idx) / bin_size # fr_100.shape = (n_trials,)

        sp_values[i], fpr, tpr = compute_spike_auc(fr_0, fr_100)
        roc_curves.append((fpr, tpr))

    mean_fpr, mean_tpr = compute_mean_roc(roc_curves) # roc_curves.shape = (n_neurons, 2)
    return sp_values, (mean_fpr, mean_tpr) # sp_values.shape = (n_neurons,)
