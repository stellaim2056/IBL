import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics import roc_auc_score

import os
import sys
sys.path.append('VISp_PSTH/v3_def')
from compute_spike_auc import compute_spike_auc
from compute_mean_roc import compute_mean_roc

import numpy as np

def compute_dp(spike_raster, trials_df, contrast_levels, start_idx, end_idx):
    """
    DP(Detection Probability) 계산 및 Mean ROC Curve 생성.
    
    Parameters:
    - spike_raster: 뉴런별 스파이크 데이터 (shape: [n_neurons, n_trials, n_bins])
    - trials_df: 트라이얼 정보 데이터프레임
    - contrast_levels: DP를 계산할 contrast level 리스트 (예: [0.0625, 0.125, 0.25])
    - start_idx: 발화율 계산 시작 인덱스
    - end_idx: 발화율 계산 끝 인덱스
    
    Returns:
    - dp_values_all: 전체 contrast를 통합한 DP 값 (뉴런별 배열)
    - mean_roc_all: 전체 contrast를 통합한 Mean ROC Curve 데이터
    """
    bin_size = 0.001
    n_neurons = spike_raster.shape[0]
    dp_values_all = np.full(n_neurons, np.nan)  # 전체 DP 저장
    roc_curves_all = []

    # 전체 contrast를 통합하기 위한 리스트
    all_fr_hit = [[] for _ in range(n_neurons)]
    all_fr_miss = [[] for _ in range(n_neurons)]

    for c in contrast_levels:
        mask_contrast = (trials_df['contrastRight'] == c)

        mask_hit = mask_contrast & (trials_df['feedbackType'] == 1)
        mask_miss = mask_contrast & (trials_df['feedbackType'] == -1)

        for i in range(n_neurons):
            fr_hit = np.sum(spike_raster[i, mask_hit, start_idx:end_idx], axis=1) / (end_idx - start_idx) / bin_size
            fr_miss = np.sum(spike_raster[i, mask_miss, start_idx:end_idx], axis=1) / (end_idx - start_idx) / bin_size

            # Hit/Miss 개수 맞추기
            n_hit, n_miss = len(fr_hit), len(fr_miss)
            if n_hit == 0 or n_miss == 0:
                continue  # 한쪽 그룹이 없으면 계산 불가

            min_trials = min(n_hit, n_miss)
            fr_hit_balanced = np.random.choice(fr_hit, min_trials, replace=False)
            fr_miss_balanced = np.random.choice(fr_miss, min_trials, replace=False)

            # Contrast별로 Hit/Miss 데이터를 저장 (나중에 합치기 위해)
            all_fr_hit[i].extend(fr_hit_balanced)
            all_fr_miss[i].extend(fr_miss_balanced)

    # Contrast 방향으로 합친 후 AUROC 계산
    for i in range(n_neurons):
        if len(all_fr_hit[i]) > 0 and len(all_fr_miss[i]) > 0:
            dp_values_all[i], fpr, tpr = compute_spike_auc(np.array(all_fr_hit[i]), np.array(all_fr_miss[i]))
            roc_curves_all.append((fpr, tpr))

    # 전체 Mean ROC Curve 계산
    mean_fpr_all, mean_tpr_all = compute_mean_roc(roc_curves_all)

    return dp_values_all, (mean_fpr_all, mean_tpr_all)


def compute_dp_contrast(spike_raster, trials_df, mask_contrast, start_idx, end_idx):
    """
    DP(Detection Probability) 계산 및 Mean ROC Curve 생성.
    """
    bin_size = 0.001
    n_neurons = spike_raster.shape[0]
    dp_values = np.full(n_neurons, np.nan)
    roc_curves = []

    mask_hit = mask_contrast & (trials_df['feedbackType'] == 1)
    mask_miss = mask_contrast & (trials_df['feedbackType'] == -1)

    for i in range(n_neurons):
        fr_hit = np.sum(spike_raster[i, mask_hit, start_idx:end_idx], axis=1) / (end_idx - start_idx) / bin_size
        fr_miss = np.sum(spike_raster[i, mask_miss, start_idx:end_idx], axis=1) / (end_idx - start_idx) / bin_size

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