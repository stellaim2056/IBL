import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from scipy.ndimage import gaussian_filter1d

from brainbox.singlecell import bin_spikes

from matplotlib import cm, colors
import os

from sub_func import save_file, _shorten_title

def plot_psth_contrast(spike_raster_all, times, trials_df,
                       left_idx, right_idx, brain_acronym, cluster_label="", 
                       save_path=None, save_title=None):
    """
    Contrast 값별 PSTH를 좌/우 자극으로 나누어 시각화
    """
    contrasts_left = np.unique(trials_df['contrastLeft'][left_idx])
    contrasts_right = np.unique(trials_df['contrastRight'][right_idx])

    def smooth(x): return gaussian_filter1d(x, sigma=100)

    psth_list_left = []
    for c in contrasts_left:
        c_mask = left_idx & (trials_df['contrastLeft'] == c)
        psth_tmp = np.nanmean(spike_raster_all[:, c_mask, :], axis=(0, 1))
        psth_list_left.append(smooth(psth_tmp))

    psth_list_right = []
    for c in contrasts_right:
        c_mask = right_idx & (trials_df['contrastRight'] == c)
        psth_tmp = np.nanmean(spike_raster_all[:, c_mask, :], axis=(0, 1))
        psth_list_right.append(smooth(psth_tmp))

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    fig.suptitle(f"[{brain_acronym}] PSTH Contrast-based ({cluster_label})")

    # Left
    for i, c_val in enumerate(contrasts_left):
        color_val = i / (max(1, len(contrasts_left) - 1))
        color = plt.cm.Reds(color_val)
        axs[0].plot(times, psth_list_left[i], label=f"{c_val*100:.1f}%", c=color)
    axs[0].axvline(0, c='k', linestyle='--')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Firing rate (Hz)')
    axs[0].set_title('Left Stimulus')
    axs[0].legend(title='Contrast')

    # Right
    for i, c_val in enumerate(contrasts_right):
        color_val = i / (max(1, len(contrasts_right) - 1))
        color = plt.cm.Blues(color_val)
        axs[1].plot(times, psth_list_right[i], label=f"{c_val*100:.1f}%", c=color)
    axs[1].axvline(0, c='k', linestyle='--')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_title('Right Stimulus')
    axs[1].legend(title='Contrast')

    plt.tight_layout()
    if save_path is None:
        save_path = "../result"
    if save_title is None:
        save_title = f"PSTH_Contrast_{brain_acronym}_{cluster_label}"
    save_file(fig, save_path, save_title)