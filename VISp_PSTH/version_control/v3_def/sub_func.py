import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from scipy.ndimage import gaussian_filter1d

import collections


from brainbox.singlecell import bin_spikes

from matplotlib import cm, colors
import os
import pandas as pd


# ----------------------------------------------------------------------
# 보조: 파일 저장 경로/이름 결정 로직
# ----------------------------------------------------------------------
def save_file(file, save_path=None, save_title=None):
    """
    file이:
      - pd.DataFrame -> CSV로 저장
      - np.ndarray   -> .npy로 저장
      - plt.Figure   -> .png 등으로 저장
      - 기타         -> 단순 print
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    if save_path is None:
        save_path = "../result"
    if save_title is None:
        save_title = "untitled"

    # 폴더 생성
    os.makedirs(save_path, exist_ok=True)
    save_filepath = os.path.join(save_path, save_title)

    if isinstance(file, pd.DataFrame):
        file.to_csv(save_filepath + ".csv")
    elif isinstance(file, np.ndarray):
        np.save(save_filepath + ".npy", file)
    elif isinstance(file, plt.Figure):
        file.savefig(save_filepath + ".png")
    elif isinstance(file, dict):
        df = pd.DataFrame(file)
        df.to_csv(save_filepath + ".csv")
    elif isinstance(file, collections.abc.Mapping): # collections import
        df = pd.DataFrame(dict(file))
        df.to_csv(save_filepath + ".csv")
    else:
        print("save_file: 알 수 없는 타입이어서 별도 저장하지 않습니다.")

        
def previous_save_file(file, save_path=None, save_title=None):
    if save_path is None:
        save_path = "../result"
    if save_title is None:
        save_title = "untitled"
    
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        save_filepath = os.path.join(save_path, save_title)
        # file이 figure일 경우
        if isinstance(file, plt.Figure):
            file.savefig(save_filepath)
        # df일 경우 csv file로 저장
        elif isinstance(file, pd.DataFrame):
            file.to_csv(save_filepath + ".csv")
            # hdf_file = os.path.join(save_path, f"{save_title}.h5")
            # file.to_hdf(hdf_file, key="df", mode="w")
        elif isinstance(file, np.ndarray):
            np.save(save_filepath, file)
        else:
            print("Invalid file type. Only figure or DataFrame is supported.")


def _shorten_title(title_str):
    """
    파일명 등에 쓰기 위해 공백/특수문자를 간단히 정리할 수도 있음.
    여기서는 예시로 스페이스만 제거 혹은 '_'로 치환 정도만.
    """
    return title_str.replace(" ", "_")

