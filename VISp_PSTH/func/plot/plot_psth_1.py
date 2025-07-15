import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from scipy.ndimage import gaussian_filter1d

from brainbox.singlecell import bin_spikes

from matplotlib import cm, colors
import os

from compute_psth import compute_psth
from sub_func import save_file, _shorten_title


# ----------------------------------------------------------------------
# 2) 4가지 PSTH 각각을 그리는 함수
#    - 단독으로 부를 때(기본 제목) vs plot_psths에서 부를 때(짧은 제목)
#    - cluster_label: "1 (single cluster)" / "10 (good cluster)" 등
#    - save_path, save_title로 그림파일 저장
# ----------------------------------------------------------------------
def plot_psth_left_vs_right(spike_raster, times,
                            left_idx, right_idx, brain_acronym, 
                            cluster_label="",
                            save_path=None,
                            save_title=None):
    """
    Left vs Right PSTH를 단독으로 그림.
    - title=None이면 기본값: "Left vs Right PSTH - {cluster_label}"
    - title!=None이면: "{title} - {cluster_label}"
    - 저장 시 파일명은 save_title+".png" (경로=save_path), 둘 다 없으면 "../result/untitled.png"
    """

    psth_left = compute_psth(spike_raster, left_idx)
    psth_right = compute_psth(spike_raster, right_idx)

    psth_left_smooth = gaussian_filter1d(psth_left, sigma=10)
    psth_right_smooth = gaussian_filter1d(psth_right, sigma=10)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, psth_left_smooth, c='crimson', label='Left')
    ax.plot(times, psth_right_smooth, c='darkblue', label='Right')
    ax.axvline(0, color='k', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(f"[{brain_acronym}] PSTH - Left vs Right ({cluster_label})")
    ax.legend()

    plt.tight_layout()
    if save_path is None:
        save_path = "../result"
    if save_title is None:
        save_title = f"PSTH_LeftRight_{brain_acronym}_{cluster_label}"
    save_file(fig, save_path, save_title)


def plot_psth_correct_vs_incorrect(spike_raster, times,
                                   correct_idx, incorrect_idx, brain_acronym, 
                                   cluster_label="",
                                   title=None,
                                   save_path=None,
                                   save_title=None):
    """
    Correct vs Incorrect PSTH
    """

    psth_correct = compute_psth(spike_raster, correct_idx)
    psth_incorrect = compute_psth(spike_raster, incorrect_idx)

    psth_correct_smooth = gaussian_filter1d(psth_correct, sigma=100)
    psth_incorrect_smooth = gaussian_filter1d(psth_incorrect, sigma=100)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, psth_correct_smooth, c='blueviolet', label='Correct')
    ax.plot(times, psth_incorrect_smooth, c='thistle', label='Incorrect')
    ax.axvline(0, color='k', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(f"[{brain_acronym}] PSTH - Correct vs Incorrect ({cluster_label})")
    ax.legend()

    plt.tight_layout()
    if save_path is None:
        save_path = "../result"
    if save_title is None:
        save_title = f"PSTH_CorrIncorr_{brain_acronym}_{cluster_label}"
    save_file(fig, save_path, save_title)


def plot_psth_left_correct_vs_left_incorrect(spike_raster, times,
                                             left_idx, correct_idx, incorrect_idx, brain_acronym, 
                                             cluster_label="",
                                             title=None,
                                             save_path=None,
                                             save_title=None):
    """
    Left correct vs Left incorrect PSTH
    """

    left_correct = left_idx & correct_idx
    left_incorrect = left_idx & incorrect_idx
    psth_lc = compute_psth(spike_raster, left_correct)
    psth_li = compute_psth(spike_raster, left_incorrect)

    psth_lc_smooth = gaussian_filter1d(psth_lc, sigma=100)
    psth_li_smooth = gaussian_filter1d(psth_li, sigma=100)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, psth_lc_smooth, c='crimson', label='Left correct')
    ax.plot(times, psth_li_smooth, c='rosybrown', label='Left incorrect')
    ax.axvline(0, color='k', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(f"[{brain_acronym}] PSTH - Left correct vs Left incorrect ({cluster_label})")
    ax.legend()

    plt.tight_layout()
    if save_path is None:
        save_path = "../result"
    if save_title is None:
        save_title = f"PSTH_LeftCorrIncorr_{brain_acronym}_{cluster_label}"
    save_file(fig, save_path, save_title)


def plot_psth_right_correct_vs_right_incorrect(spike_raster, times,
                                               right_idx, correct_idx, incorrect_idx, brain_acronym, 
                                               cluster_label="",
                                               title=None,
                                               save_path=None,
                                               save_title=None):
    """
    Right correct vs Right incorrect PSTH
    """

    right_correct = right_idx & correct_idx
    right_incorrect = right_idx & incorrect_idx
    psth_rc = compute_psth(spike_raster, right_correct)
    psth_ri = compute_psth(spike_raster, right_incorrect)

    psth_rc_smooth = gaussian_filter1d(psth_rc, sigma=100)
    psth_ri_smooth = gaussian_filter1d(psth_ri, sigma=100)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(times, psth_rc_smooth, c='darkblue', label='Right correct')
    ax.plot(times, psth_ri_smooth, c='slategray', label='Right incorrect')
    ax.axvline(0, color='k', linestyle='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Firing rate (Hz)')
    ax.set_title(f"[{brain_acronym}] PSTH - Rigth correct vs Right incorrect ({cluster_label})")
    ax.legend()

    plt.tight_layout()
    if save_path is None:
        save_path = "../result"
    if save_title is None:
        save_title = f"PSTH_RightCorrIncorr_{brain_acronym}_{cluster_label}"
    save_file(fig, save_path, save_title)


# ----------------------------------------------------------------------
# 3) 4개 PSTH 한번에 그리는 함수

def plot_psths_each4(spike_raster, times,
               left_idx, right_idx, correct_idx, incorrect_idx, brain_acronym, 
                cluster_label="",
                save_path=None):
    # 1) Left vs Right
    plot_psth_left_vs_right(
        spike_raster, times, left_idx, right_idx, brain_acronym,
        cluster_label=cluster_label,
        save_path=save_path,
    )
    # 2) Correct vs Incorrect
    plot_psth_correct_vs_incorrect(
        spike_raster, times, correct_idx, incorrect_idx, brain_acronym,
        cluster_label=cluster_label,
        save_path=save_path,
    )
    # 3) Left correct vs Left incorrect
    plot_psth_left_correct_vs_left_incorrect(
        spike_raster, times, left_idx, correct_idx, incorrect_idx, brain_acronym,
        cluster_label=cluster_label,
        save_path=save_path,
    )
    # 4) Right correct vs Right incorrect
    plot_psth_right_correct_vs_right_incorrect(
        spike_raster, times, right_idx, correct_idx, incorrect_idx, brain_acronym,
        cluster_label=cluster_label,
        save_path=save_path,
    )