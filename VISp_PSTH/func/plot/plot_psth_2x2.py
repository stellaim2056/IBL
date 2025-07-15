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
# 3) 4가지 PSTH를 2x2 한 그림에 통합하여 표시하고, 
#    - 각 PSTH 함수를 "짧은 제목"으로도 호출하여 별도로 저장
#    - 2x2 그림 자체도 저장
# ----------------------------------------------------------------------

def plot_psth_2x2(spike_raster, times,
               left_idx, right_idx, correct_idx, incorrect_idx, brain_acronym, 
               cluster_label="", 
               save_path=None, save_title=None):
    """
    2x2 subplots:
      (1) Left vs Right
      (2) Correct vs Incorrect
      (3) Left correct vs Left incorrect
      (4) Right correct vs Right incorrect

    - cluster_label: 예) "1 (single cluster)", "5 (good cluster)", "30 (all cluster)"
    - main_title: 2x2 전체 그림의 메인 제목
    - 각 서브플롯은 짧은 소제목("Left/Right", "Corr/Incorr", ...)
    - save_path, save_title가 지정되지 않으면 "../result/untitled.png" 로 저장
    - 4개 PSTH (짧은 제목)도 각각 별도 figure로 자동 저장
      => single PSTH 함수 호출할 때 save_path, save_title 변경
    """

    # 이제 2x2 서브플롯 그림
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    fig.suptitle(f"[{brain_acronym}] PSTH ({cluster_label})")

    # (0,0): Left vs Right
    _plot_sub_psth_left_vs_right(axs[0, 0], times, spike_raster,
                                 left_idx, right_idx)
    # (0,1): Correct vs Incorrect
    _plot_sub_psth_correct_vs_incorrect(axs[0, 1], times, spike_raster,
                                        correct_idx, incorrect_idx)
    # (1,0): Left correct vs Left incorrect
    _plot_sub_psth_left_correct_vs_left_incorrect(axs[1, 0], times, spike_raster,
                                                  left_idx, correct_idx, incorrect_idx)
    # (1,1): Right correct vs Right incorrect
    _plot_sub_psth_right_correct_vs_right_incorrect(axs[1, 1], times, spike_raster,
                                                    right_idx, correct_idx, incorrect_idx)

    for row in axs:
        for ax in row:
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Firing rate (Hz)')

    plt.tight_layout()
    if save_path is None:
        save_path = "../result"
    if save_title is None:
        save_title = f"PSTH_2x2_{brain_acronym}_{cluster_label}"
    save_file(fig, save_path, save_title)


# ----------------------------------------------------------------------
# 내부에서만 사용하는 sub-plot용 함수 (2x2 배치용, fig/ax 분리)
# : "짧은 소제목" 버전
# ----------------------------------------------------------------------
def _plot_sub_psth_left_vs_right(ax, times, spike_raster,
                                 left_idx, right_idx):
    psth_left = compute_psth(spike_raster, left_idx)
    psth_right = compute_psth(spike_raster, right_idx)
    ax.plot(times, gaussian_filter1d(psth_left, 100), c='crimson', label='Left')
    ax.plot(times, gaussian_filter1d(psth_right, 100), c='darkblue', label='Right')
    ax.axvline(0, c='k', linestyle='--')
    ax.set_title("Left/Right")
    ax.legend()

def _plot_sub_psth_correct_vs_incorrect(ax, times, spike_raster,
                                        correct_idx, incorrect_idx):
    psth_correct = compute_psth(spike_raster, correct_idx)
    psth_incorrect = compute_psth(spike_raster, incorrect_idx)
    ax.plot(times, gaussian_filter1d(psth_correct, 100), c='blueviolet', label='Correct')
    ax.plot(times, gaussian_filter1d(psth_incorrect, 100), c='thistle', label='Incorrect')
    ax.axvline(0, c='k', linestyle='--')
    ax.set_title("Corr/Incorr")
    ax.legend()

def _plot_sub_psth_left_correct_vs_left_incorrect(ax, times, spike_raster,
                                                  left_idx, correct_idx, incorrect_idx):
    left_correct = left_idx & correct_idx
    left_incorrect = left_idx & incorrect_idx
    psth_lc = compute_psth(spike_raster, left_correct)
    psth_li = compute_psth(spike_raster, left_incorrect)
    ax.plot(times, gaussian_filter1d(psth_lc, 100), c='crimson', label='Left correct')
    ax.plot(times, gaussian_filter1d(psth_li, 100), c='rosybrown', label='Left incorrect')
    ax.axvline(0, c='k', linestyle='--')
    ax.set_title("LeftCorr/Incorr")
    ax.legend()

def _plot_sub_psth_right_correct_vs_right_incorrect(ax, times, spike_raster,
                                                    right_idx, correct_idx, incorrect_idx):
    right_correct = right_idx & correct_idx
    right_incorrect = right_idx & incorrect_idx
    psth_rc = compute_psth(spike_raster, right_correct)
    psth_ri = compute_psth(spike_raster, right_incorrect)
    ax.plot(times, gaussian_filter1d(psth_rc, 100), c='darkblue', label='Right correct')
    ax.plot(times, gaussian_filter1d(psth_ri, 100), c='slategray', label='Right incorrect')
    ax.axvline(0, c='k', linestyle='--')
    ax.set_title("RightCorr/Incorr")
    ax.legend()

