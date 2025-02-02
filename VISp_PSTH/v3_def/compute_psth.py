import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from scipy.ndimage import gaussian_filter1d

from brainbox.singlecell import bin_spikes

from matplotlib import cm, colors
import os



def compute_psth(spike_raster, trial_mask):
    """
    spike_raster: shape = (nUnits, nTrials, nBins)
    trial_mask: shape=(nTrials,) boolean mask
    -> PSTH (nBins,) : (nUnits, trials) 방향으로 평균
    """
    return np.nanmean(spike_raster[:, trial_mask, :], axis=(0, 1))

