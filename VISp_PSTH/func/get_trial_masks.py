import numpy as np

def get_trial_masks(trials_df):
    """
    trials 데이터를 받아서
    left_idx, right_idx, correct_idx, incorrect_idx 등을 반환.
    """
    left_idx = ~np.isnan(trials_df['contrastLeft'])
    right_idx = ~np.isnan(trials_df['contrastRight'])
    correct_idx = (trials_df['feedbackType'] == 1)
    incorrect_idx = (trials_df['feedbackType'] == -1)
    return left_idx, right_idx, correct_idx, incorrect_idx
