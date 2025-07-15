import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sub_func import save_file
from compute_binned_stats import compute_binned_stats

def plot_sp(sp_values, mean_fr_values, 
              plot_info, save_path=None, save_title=None):
    """
    SP vs Mean Firing Rate 그래프 생성.
    """
    brain_acronym, cluster_type, end_time = plot_info

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(sp_values, mean_fr_values, c='blue', edgecolors=None, alpha=0.3, s=10)
    ax.set_xlabel(f"SP SO+{end_time}s")
    ax.set_ylabel("Mean Firing Rate (Hz)")
    ax.set_title(f"[{brain_acronym}] SP (AUC, 0% vs. 100%) - {cluster_type.capitalize()} clusters")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(0.5)) ## x값이 5의 배수일 때마다 메인 눈금 표시
    ax.xaxis.set_major_formatter('{x:.1f}') ## 메인 눈금이 표시될 형식
    ax.xaxis.set_minor_locator(MultipleLocator(0.1)) ## 서브 눈금은 x값이 1의 배수인 경우마다 표시

    plt.tight_layout()
    if save_path is None:
        save_path = "../result"
    if save_title is not None:
        save_file(fig, save_path, save_title)
    else:
        save_file(fig, save_path, f"SP_SO+{end_time}s_{cluster_type}")

def plot_sp_transformed(sp_values, mean_fr_values, 
                        plot_info, save_path=None, save_title=None):
    
    brain_acronym, cluster_type, end_time = plot_info

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"[{brain_acronym}] SP SO+{end_time}s (AUC, 0% vs. 100%) - {cluster_type.capitalize()} clusters")

    # SP Scatter Plot (원본)
    axs[0].scatter(sp_values, mean_fr_values, c='blue', edgecolors=None, alpha=0.3, s=10)
    # SP 오차 막대 추가
    bin_centers, means, y_errs, x_errs = compute_binned_stats(sp_values, mean_fr_values)
    axs[0].errorbar(bin_centers, means, c='blue', xerr=x_errs, yerr=y_errs)
    axs[0].plot(bin_centers, means, c='blue', linestyle='-', linewidth=1)  # 선 추가
    axs[0].set_xlabel(f"SP SO+{end_time}s")
    axs[0].set_ylabel("Mean Firing Rate (Hz)")
    axs[0].set_xlim(0, 1)
    axs[0].xaxis.set_major_locator(MultipleLocator(0.5)) ## 메인 눈금
    axs[0].xaxis.set_major_formatter('{x:.1f}') ## 메인 눈금이 표시될 형식
    axs[0].xaxis.set_minor_locator(MultipleLocator(0.1)) ## 서브 눈금]

    # SP Scatter Plot (변환)
    sp_transformed = 2 * np.abs(sp_values - 0.5)
    axs[1].scatter(sp_transformed, mean_fr_values, c='blue', edgecolors=None, alpha=0.3, s=10)
    # SP 오차 막대 추가
    bin_centers, means, y_errs, x_errs = compute_binned_stats(sp_transformed, mean_fr_values)
    axs[1].errorbar(bin_centers, means, c='blue', xerr=x_errs, yerr=y_errs)
    axs[1].plot(bin_centers, means, c='blue', linestyle='-', linewidth=1)  # 선 추가
    axs[1].set_xlabel(f"2*abs(SP-0.5) SO+{end_time}s")
    axs[1].set_ylabel("Mean Firing Rate (Hz)")
    axs[1].xaxis.set_major_locator(MultipleLocator(0.2)) ## 메인 눈금
    axs[1].xaxis.set_major_formatter('{x:.1f}') ## 메인 눈금이 표시될 형식
    axs[1].xaxis.set_minor_locator(MultipleLocator(0.1)) ## 서브 눈금]

    plt.tight_layout()
    if save_path is None:
        save_path = "../result"
    if save_title is not None:
        save_file(fig, save_path, save_title)
    else:
        save_file(fig, save_path, f"SP_transformed_SO+{end_time}s_{cluster_type}")

def plot_sp_roc(sp_values, mean_roc_sp, mean_fr_values, 
                plot_info, save_path=None, save_title=None):

    brain_acronym, cluster_type, end_time = plot_info

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"[{brain_acronym}] SP (AUC, 0% vs. 100%) - {cluster_type.capitalize()} clusters")

    # SP ROC Curve
    axs[0].plot(mean_roc_sp[0], mean_roc_sp[1], 'b-', label="Mean SP ROC")
    axs[0].plot([0,1], [0,1], '--', color='gray', label='Chance')
    axs[0].set_title("Mean ROC Curve for SP")
    axs[0].set_xlabel("False Positive Rate")
    axs[0].set_ylabel("True Positive Rate")
    axs[0].legend()

    # SP Scatter Plot
    axs[1].scatter(sp_values, mean_fr_values, c='blue', edgecolors=None, alpha=0.3, s=10)
    axs[1].set_xlabel(f"SP SO+{end_time}s")
    axs[1].set_ylabel("Mean Firing Rate (Hz)")
    axs[1].set_title("SP vs Mean Firing Rate")
    axs[1].set_xlim(0, 1)
    axs[1].xaxis.set_major_locator(MultipleLocator(0.5)) ## x값이 5의 배수일 때마다 메인 눈금 표시
    axs[1].xaxis.set_major_formatter('{x:.1f}') ## 메인 눈금이 표시될 형식
    axs[1].xaxis.set_minor_locator(MultipleLocator(0.1)) ## 서브 눈금은 x값이 1의 배수인 경우마다 표시

    plt.tight_layout()
    if save_path is None:
        save_path = "../result"
    if save_title is not None:
        save_file(fig, save_path, save_title)
    else:
        save_file(fig, save_path, f"SP_roc_SO+{end_time}s_{cluster_type}")