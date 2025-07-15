import numpy as np

# -----------------------------------------------------------------------------
# 0. 유틸리티 함수들
# -----------------------------------------------------------------------------
def print_cluster_info(clusters, clusters_good, brain_acronym):
    """
    주어진 클러스터 dict(clusters)와 good 클러스터 dict(clusters_good)에 대해
    각 영역(Acronym) 별 클러스터 수와 'brain_acronym'과 관련된 클러스터 수를 출력합니다.
    """
    # 전체 클러스터 및 good 클러스터 개수
    all_clusters = clusters['label'].shape[0]
    good_clusters = clusters_good['label'].shape[0]

    print(f'\nNo. of clusters in the session (good | all): [{good_clusters} | {all_clusters}]')

    # 각 영역(Acronym)에서 클러스터 통계
    acronyms_all = clusters['acronym']
    acronyms_good = clusters_good['acronym']
    unique_acronyms_all, count_all = np.unique(acronyms_all, return_counts=True)
    unique_acronyms_good, count_good = np.unique(acronyms_good, return_counts=True)

    # VISp와 관련된 클러스터 개수 초기화
    num_all_clusters = 0
    num_good_clusters = 0

    # 각 영역별 클러스터 수 출력
    print(f'\nNo. of clusters in each region:')
    for region in unique_acronyms_all:
        # 모든 클러스터에서 해당 영역의 수
        total_count = count_all[np.where(unique_acronyms_all == region)][0]
        # good 클러스터에서 해당 영역의 수
        good_count = count_good[np.where(unique_acronyms_good == region)][0] if region in unique_acronyms_good else 0
    
        print(f'{region}: {good_count} | {total_count}')
        
        # VISp와 관련된 클러스터 합산
        if brain_acronym in region:
            num_all_clusters += total_count
            num_good_clusters += good_count

    # VISp와 관련된 클러스터 출력
    print(f'\nNo. of clusters in {brain_acronym} (good | all): [{num_good_clusters} | {num_all_clusters}]')


def previous_print_cluster_info(clusters, brain_acronym, label_description="All"):
    """
    주어진 클러스터 dict(clusters 또는 clusters_good)에 대해
    뇌영역 별 개수를 출력하고, 'brain_acronym' 문자열이 포함된 영역의 클러스터 수도 요약해줍니다.
    """
    acronyms = clusters['acronym']
    unique_acr, counts = np.unique(acronyms, return_counts=True)

    print(f"\n-- {label_description} clusters info --")
    print(f"Total {label_description} clusters: {clusters['label'].shape[0]}")
    print(f"Cluster count by region:\n{'='*40}")

    n_in_region = 0
    for acr, c in zip(unique_acr, counts):
        print(f" {acr}: {c}")
        if brain_acronym in acr:
            n_in_region += c

    print(f"\n=> Clusters with '{brain_acronym}' in acronym: {n_in_region}\n")


