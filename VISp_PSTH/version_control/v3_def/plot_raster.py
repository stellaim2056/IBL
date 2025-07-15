import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

from brainbox.singlecell import bin_spikes
import os

from sub_func import save_file

def plot_raster(spike_raster, times, events, brain_acronym, cluster_label, save_path=None, save_title=None):
    """
    spike_raster: shape = [nUnits, nTrials, nBins]
      - nUnits=1이면 'Single Cluster' raster imshow (trial x time)
      - nUnits>1이면 'Multi-Cluster' raster 2가지 방식 (trial별 평균, cluster별 평균)
    """
    nUnits, nTrials, nBins = spike_raster.shape

    if nUnits == 1:
        # Single cluster -> trial x time bins imshow
        spike_raster_2d = spike_raster[0]  # shape=(nTrials, nBins)
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(spike_raster_2d,
                       extent=[times[0], times[-1], 0, nTrials],
                       origin='lower', cmap='binary', aspect='auto')
        ax.axvline(0, color='k', linestyle='--')
        ax.set_xlabel('Time from stimulus (s)')
        ax.set_ylabel('Trial index')
        ax.set_title(f"[{brain_acronym}] Spike Raster ({cluster_label})")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Firing rate (Hz)')
        plt.tight_layout()

    else:
        # Multiple clusters -> (좌) trial별 평균, (우) cluster별 평균
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"[{brain_acronym}] Spike Raster ({cluster_label})")
        
        # (좌) trial별 평균 (=> shape=(nTrials, nBins))
        mean_over_clusters = np.nanmean(spike_raster, axis=0)
        im1 = axs[0].imshow(mean_over_clusters,
                            extent=[times[0], times[-1], 0, nTrials],
                            origin='lower', cmap='binary', aspect='auto')
        axs[0].axvline(0, c='k', linestyle='--')
        axs[0].set_xlabel('Time from stimulus (s)')
        axs[0].set_ylabel('Trial index')
        axs[0].set_title('Mean over clusters')
        plt.colorbar(im1, ax=axs[0], label='Firing rate (Hz)')

        # (우) 클러스터별 평균 (=> shape=(nUnits, nBins))
        mean_over_trials = np.nanmean(spike_raster, axis=1)
        im2 = axs[1].imshow(mean_over_trials,
                            extent=[times[0], times[-1], 0, nUnits],
                            origin='lower', cmap='binary', aspect='auto')
        axs[1].axvline(0, c='k', linestyle='--')
        axs[1].set_xlabel('Time from stimulus (s)')
        axs[1].set_ylabel('Cluster index')
        axs[1].set_title('Mean over trials')
        plt.colorbar(im2, ax=axs[1], label='Firing rate (Hz)')

        plt.tight_layout()
  

    if save_path is None:
        save_path = "../result"
    if save_title is None:
        save_title = f"Spike_raster_{brain_acronym}_{cluster_label}"
    save_file(fig, save_path, save_title)