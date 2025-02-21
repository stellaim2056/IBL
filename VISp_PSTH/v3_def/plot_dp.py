import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from sub_func import save_file
from compute_binned_stats import compute_binned_stats

def plot_dp(dp_values_dict, mean_fr_values, 
            plot_info, save_path=None, save_title=None):
    """
    DP vs Mean Firing Rate 그래프 생성.
    """
    brain_acronym, cluster_type, end_time, contrast_levels, colors = plot_info

    fig, ax = plt.subplots(figsize=(6, 5))
    for contrast, color in zip(contrast_levels, colors):
        ax.scatter(dp_values_dict[contrast], mean_fr_values, c=color, edgecolors=None, alpha=0.3, s=10, label=f"{contrast*100}%")
    ax.legend(title="Contrast Level")
    ax.set_xlabel(f"DP SO+{end_time}s")
    ax.set_ylabel("Mean Firing Rate (Hz)")
    ax.set_title(f"[{brain_acronym}] DP (AUC, hit vs miss) - {cluster_type.capitalize()} clusters")

    ax.xaxis.set_major_locator(MultipleLocator(0.5)) ## 메인 눈금
    ax.xaxis.set_major_formatter('{x:.1f}') ## 메인 눈금이 표시될 형식
    ax.xaxis.set_minor_locator(MultipleLocator(0.1)) ## 서브 눈금]
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = "../result"
    if save_title is not None:
        save_file(fig, save_path, save_title)
    else:
        save_file(fig, save_path, f"DP_SO+{end_time}s_{cluster_type}")



def plot_dp_transformed(dp_values_dict, mean_fr_values, 
                        plot_info, save_path=None, save_title=None):
    brain_acronym, cluster_type, end_time, contrast_levels, colors = plot_info
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"[{brain_acronym}] DP SO+{end_time}s (AUC, hit vs miss) - {cluster_type.capitalize()} clusters")
    
    # DP Scatter Plot (원본)
    for contrast, color in zip(contrast_levels, colors):
        axs[0].scatter(dp_values_dict[contrast], mean_fr_values, c=color, edgecolors=None, alpha=0.3, s=10, label=f"{contrast*100}%")
        # DP 오차 막대 추가
        bin_centers, means, y_errs, x_errs = compute_binned_stats(dp_values_dict[contrast], mean_fr_values)
        axs[0].errorbar(bin_centers, means, c=color, xerr=x_errs, yerr=y_errs)
        axs[0].plot(bin_centers, means, c=color, linestyle='-', linewidth=1)  # 선 추가
    axs[0].legend(title="Contrast Level")
    axs[0].set_xlabel(f"DP SO+{end_time}s")
    axs[0].set_ylabel("Mean Firing Rate (Hz)")
    axs[0].xaxis.set_major_locator(MultipleLocator(0.5)) ## 메인 눈금
    axs[0].xaxis.set_major_formatter('{x:.1f}') ## 메인 눈금이 표시될 형식
    axs[0].xaxis.set_minor_locator(MultipleLocator(0.1)) ## 서브 눈금]

    # DP Scatter Plot (변환)
    for contrast, color in zip(contrast_levels, colors):
        dp_transformed = 2 * np.abs(dp_values_dict[contrast] - 0.5)
        axs[1].scatter(dp_transformed, mean_fr_values, c=color, edgecolors=None, alpha=0.3, s=10, label=f"{contrast*100}%")
        # DP 오차 막대 추가
        bin_centers, means, y_errs, x_errs = compute_binned_stats(dp_transformed, mean_fr_values)
        axs[1].errorbar(bin_centers, means, c=color, xerr=x_errs, yerr=y_errs)
        axs[1].plot(bin_centers, means, c=color, linestyle='-', linewidth=1)  # 선 추가
    axs[1].legend(title="Contrast Level")
    axs[1].set_xlabel(f"2*abs(DP-0.5) SO+{end_time}s")
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
        save_file(fig, save_path, f"DP_transformed_SO+{end_time}s_{cluster_type}")

def plot_dp_roc(dp_values_dict, mean_roc_curves_dict, mean_fr_values, 
                plot_info, save_path=None, save_title=None):
    brain_acronym, cluster_type, end_time, contrast_levels, colors = plot_info
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"[{brain_acronym}] DP (AUC, hit vs miss) - {cluster_type.capitalize()} clusters")

    for contrast, color in zip(contrast_levels, colors):
        axs[0].plot(mean_roc_curves_dict[contrast][0], mean_roc_curves_dict[contrast][1], color=color, label=f"{contrast*100}% DP ROC")
    axs[0].plot([0,1], [0,1], '--', color='gray', label='Chance')
    axs[0].set_title("Mean ROC Curve for DP")
    axs[0].set_xlabel("False Positive Rate")
    axs[0].set_ylabel("True Positive Rate")
    axs[0].legend(title="Contrast Level")

    # DP Scatter Plot (강도별)
    for contrast, color in zip(contrast_levels, colors):
        axs[1].scatter(dp_values_dict[contrast], mean_fr_values, c=color, edgecolors=None, alpha=0.3, s=10, label=f"{contrast*100}%")
    axs[1].legend(title="Contrast Level")
    axs[1].set_xlabel(f"DP SO+{end_time}s")
    axs[1].set_ylabel("Mean Firing Rate (Hz)")
    axs[1].set_title("DP vs Mean Firing Rate")
    axs[1].xaxis.set_major_locator(MultipleLocator(0.5)) ## 메인 눈금
    axs[1].xaxis.set_major_formatter('{x:.1f}') ## 메인 눈금이 표시될 형식
    axs[1].xaxis.set_minor_locator(MultipleLocator(0.1)) ## 서브 눈금]
    
    plt.tight_layout()
    if save_path is None:
        save_path = "../result"
    if save_title is not None:
        save_file(fig, save_path, save_title)
    else:
        save_file(fig, save_path, f"DP_roc_SO+{end_time}s_{cluster_type}")