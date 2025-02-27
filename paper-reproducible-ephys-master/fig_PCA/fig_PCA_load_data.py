import pandas as pd
import numpy as np
from reproducible_ephys_functions import save_data_path


def load_dataframe(exists_only=False):
    df_path = save_data_path(figure='fig_PCA').joinpath('fig_PCA_dataframe.csv')
    if exists_only:
        return df_path.exists()
    else:
        if df_path.exists():
            return pd.read_csv(df_path)
        else:
            return None


def load_data(event='move', split='rt', norm='z_score', smoothing=None, exists_only=False):

    smoothing = smoothing or None
    norm = norm or None

    df_path = save_data_path(figure='fig_PCA').joinpath(
        f'fig_PCA_data_event_{event}_split_{split}_smoothing_{smoothing}_norm_{norm}.npz')
    if exists_only:
        return df_path.exists()
    else:
        if df_path.exists():
            return dict(np.load(df_path, allow_pickle=True))
        else:
            return None
