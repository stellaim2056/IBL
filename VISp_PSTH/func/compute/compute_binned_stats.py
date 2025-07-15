import numpy as np

def compute_binned_stats(x_values, y_values, num_bins=10):
    """
    x축 구간을 각 구간마다 동일한 데이터 포인트 수를 갖도록 나눈 후,
    각 bin 내부의 (x 평균, y 평균, y값 SEM, x값 SEM)을 계산하는 함수.

    Parameters
    ----------
    x_values : array-like
        x 좌표에 해당하는 1차원 배열
    y_values : array-like
        y 좌표(측정 값)에 해당하는 1차원 배열
    num_bins : int, optional
        나눌 구간(bin)의 개수 (기본값 = 10)

    Returns
    -------
    bin_centers : np.ndarray
        각 bin 내부의 x 평균 (bin별 대표 x 값)
    means : np.ndarray
        각 bin 내부의 y 평균
    y_errs : np.ndarray
        각 bin 내부 y값의 표준 오차(Standard Error of the Mean)
    x_errs : np.ndarray
        각 bin 내부 x값의 표준 오차(필요치 않으면 사용하지 않아도 됨)
    """

    # 1) x 기준으로 정렬
    sorted_indices = np.argsort(x_values)
    x_sorted = x_values[sorted_indices]
    y_sorted = y_values[sorted_indices]
    
    # 2) bin 개수와 bin 크기 설정 (각 bin마다 동일한 data 수)
    n = len(x_sorted)
    bin_size = int(np.ceil(n / num_bins))  # 올림으로 구간을 나눔
    
    bin_centers = []
    means = []
    y_errs = []
    x_errs = []
    
    # 3) bin별 통계치 계산
    for i in range(num_bins):
        start = i * bin_size
        end = min((i + 1) * bin_size, n)
        
        # 혹시 마지막 bin에서 데이터가 부족할 수 있으므로 예외처리
        if start >= n:
            break
        
        bin_x = x_sorted[start:end]
        bin_y = y_sorted[start:end]
        
        # bin 내부 x,y 평균
        mean_x = np.mean(bin_x)
        mean_y = np.mean(bin_y)
        
        # 표준 오차(SEM) = 표준편차 / sqrt(n)
        sem_y = np.std(bin_y, ddof=1) / np.sqrt(len(bin_y))
        sem_x = np.std(bin_x, ddof=1) / np.sqrt(len(bin_x))

        # # (기존 SEM 계산 대신) 표준편차로 계산
        # std_y = np.std(bin_y, ddof=1)
        # std_x = np.std(bin_x, ddof=1)
        
        bin_centers.append(mean_x)
        means.append(mean_y)
        y_errs.append(sem_y)
        x_errs.append(sem_x)
        
    return (
        np.array(bin_centers), 
        np.array(means), 
        np.array(y_errs), 
        np.array(x_errs)
    )


def previous0_compute_binned_stats(x_values, y_values, num_bins=20):
    """
    주어진 x_values를 num_bins 개의 구간으로 나누되,
    각 구간에 동일한 데이터 포인트 수가 들어가도록 설정하고,
    x_values와 y_values의 평균 및 표준 오차(SE)를 계산.

    Parameters:
    - x_values: X축 데이터 (SP 또는 DP)
    - y_values: Y축 데이터 (Mean Firing Rate)
    - num_bins: 구간 개수 (default=5)

    Returns:
    - bin_centers: 각 구간의 중앙값 (x_values의 중앙값)
    - means: 각 구간의 y_values 평균
    - y_errs: y_values의 표준 오차(SE)
    - x_errs: x_values의 표준 오차(SE)
    """
    sorted_indices = np.argsort(x_values)
    x_sorted = x_values[sorted_indices]
    y_sorted = y_values[sorted_indices]

    num_points = len(x_values)
    points_per_bin = num_points // num_bins  # 각 bin에 들어갈 데이터 수

    bin_centers, means, y_errs, x_errs = [], [], [], []

    for i in range(num_bins):
        start_idx = i * points_per_bin
        end_idx = (i + 1) * points_per_bin if i < num_bins - 1 else num_points

        bin_x_values = x_sorted[start_idx:end_idx]
        bin_y_values = y_sorted[start_idx:end_idx]

        # 중앙값 및 평균 계산
        bin_centers.append(np.median(bin_x_values))
        means.append(np.mean(bin_y_values))

        # 표준 오차(SE) 계산
        y_err = np.std(bin_y_values) / np.sqrt(len(bin_y_values))
        x_err = np.std(bin_x_values) / np.sqrt(len(bin_x_values))

        y_errs.append(y_err)
        x_errs.append(x_err)

    return np.array(bin_centers), np.array(means), np.array(y_errs), np.array(x_errs)



def previous1_compute_binned_stats(x_values, y_values, num_bins=5, use_se=True):
    """
    주어진 x_values를 num_bins 개의 구간으로 나누어,
    각 구간에서 y_values의 평균과 표준편차(SE 또는 SD)를 계산하고,
    x_values의 표준편차(SE 또는 SD)를 계산하여 대칭적인 오차 막대를 추가할 수 있도록 한다.
    
    Parameters:
    - x_values: X축 데이터 (SP 또는 DP)
    - y_values: Y축 데이터 (Mean Firing Rate)
    - num_bins: 구간 개수 (default=5)
    - use_se: True이면 표준 오차(SE), False이면 표준 편차(SD) 사용

    Returns:
    - bin_centers: 각 구간의 중앙값
    - means: 각 구간의 평균
    - y_errs: Y축 값의 대칭 오차
    - x_errs: X축 값의 대칭 오차
    """
    bins = np.linspace(np.min(x_values), np.max(x_values), num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    means, y_errs, x_errs = [], [], []

    for i in range(len(bins) - 1):
        bin_mask = (x_values >= bins[i]) & (x_values < bins[i + 1])
        if np.sum(bin_mask) > 0:  # 빈 구간이 아닌 경우만 포함
            bin_x_values = x_values[bin_mask]
            bin_y_values = y_values[bin_mask]

            # 평균 계산
            means.append(np.mean(bin_y_values))

            # 표준 오차(SE) 또는 표준 편차(SD) 계산
            if use_se:
                y_err = np.std(bin_y_values) / np.sqrt(np.sum(bin_mask))  # 표준 오차(SE)
                x_err = np.std(bin_x_values) / np.sqrt(np.sum(bin_mask))  # 표준 오차(SE)
            else:
                y_err = np.std(bin_y_values)  # 표준 편차(SD)
                x_err = np.std(bin_x_values)  # 표준 편차(SD)

            y_errs.append(y_err)
            x_errs.append(x_err)
        else:
            means.append(np.nan)
            y_errs.append(np.nan)
            x_errs.append(np.nan)

    return bin_centers, means, np.array(y_errs), np.array(x_errs)  # 대칭 오차 반환


def previous2_compute_binned_stats(x_values, y_values, num_bins=5):
    """
    주어진 x_values를 num_bins 개의 구간으로 나누어,
    각 구간에서 y_values의 평균과 비대칭 표준편차(SE)를 계산하고,
    x_values의 비대칭 표준편차도 계산하여 가로 오차 막대를 추가할 수 있도록 한다.
    
    Returns:
    - bin_centers: 각 구간의 중앙값
    - means: 각 구간의 평균
    - y_errs: Y축 값의 비대칭 오차 [하한, 상한]
    - x_errs: X축 값의 비대칭 오차 [하한, 상한]
    """
    bins = np.linspace(np.min(x_values), np.max(x_values), num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    means, y_errs, x_errs = [], [], []

    for i in range(len(bins) - 1):
        bin_mask = (x_values >= bins[i]) & (x_values < bins[i + 1])
        if np.sum(bin_mask) > 0:  # 빈 구간이 아닌 경우만 포함
            bin_y_values = y_values[bin_mask]
            means.append(np.mean(bin_y_values))

            # 비대칭 Y축 오차
            y_err_lower = np.mean(bin_y_values) - np.min(bin_y_values)
            y_err_upper = np.max(bin_y_values) - np.mean(bin_y_values)
            y_errs.append([y_err_lower, y_err_upper])

            # 비대칭 X축 오차
            bin_x_values = x_values[bin_mask]
            x_err_lower = np.mean(bin_x_values) - np.min(bin_x_values)
            x_err_upper = np.max(bin_x_values) - np.mean(bin_x_values)
            x_errs.append([x_err_lower, x_err_upper])
        else:
            means.append(np.nan)
            y_errs.append([np.nan, np.nan])
            x_errs.append([np.nan, np.nan])

    return bin_centers, means, np.array(y_errs).T, np.array(x_errs).T  # (2, N) 형태로 변환


