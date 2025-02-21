import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def compute_spike_auc(spike_rates_group1, spike_rates_group2): 
    """
    두 그룹의 발화율을 사용하여 AUROC 값을 계산하고 ROC Curve 데이터 생성.
    """
    if len(spike_rates_group1) == 0 or len(spike_rates_group2) == 0:
        return np.nan, [], []  

    labels = np.concatenate([np.zeros_like(spike_rates_group1), np.ones_like(spike_rates_group2)])
    scores = np.concatenate([spike_rates_group1, spike_rates_group2])

    fpr, tpr, _ = roc_curve(labels, scores)
    auc_value = roc_auc_score(labels, scores)

    return auc_value, fpr, tpr

