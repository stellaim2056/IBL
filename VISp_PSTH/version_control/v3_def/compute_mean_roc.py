import numpy as np
from scipy.interpolate import interp1d

def compute_mean_roc(roc_curves):
    """
    여러 개의 ROC Curve 데이터를 받아 평균 ROC Curve를 계산.
    """
    all_fprs = np.linspace(0, 1, 100)  # 고정된 FPR 값 (공통 기준)
    interpolated_tprs = []

    for fpr, tpr in roc_curves:
        if len(fpr) > 0 and len(tpr) > 0:
            interp_func = interp1d(fpr, tpr, kind='linear', bounds_error=False, fill_value=0)
            interpolated_tprs.append(interp_func(all_fprs))

    if len(interpolated_tprs) > 0:
        mean_tpr = np.mean(interpolated_tprs, axis=0)  # TPR 평균
        return all_fprs, mean_tpr
    else:
        return [], []