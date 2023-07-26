from typing import List, Dict

import pandas as pd
import numpy as np
import torch

from pytorch_utils.data_loaders.TemporalEmbeddingsLoader import TemporalEmbeddingsLoader


def divide_df_on_videos(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    unique_video_names = df['video_name'].unique()
    result = {}
    for video_name in unique_video_names:
        result[video_name] = df[df['video_name'] == video_name]
    return result


def load_data(paths_to_dfs: List[str]) -> List[Dict[str, pd.DataFrame]]:
    # load dfs
    dfs = []
    for path_to_df in paths_to_dfs:
        dfs.append(pd.read_csv(path_to_df))
    # add columns "timestep" to all dfs
    for df_idx in range(len(dfs)):
        dfs[df_idx]['timestep'] = dfs[df_idx]['frame_num']*0.2
        # reorder columns
        new_order = ['video_name', 'frame_num', 'timestep', 'q1', 'q2', 'q3'] + [f'emb_{i}' for i in range(256)]

    # the data format of the df is the following:
    # video_name, frame_num, q1, q2, q3, emb_0, emb_1, ..., emb_255
    # we first need to divide the data on videos so that every video will be represented as Dict[str, pd.DataFrame]
    # DataFrame has the same columns: video_name, frame_num, q1, q2, q3, emb_0, emb_1, ..., emb_255
    # str is the name of the video
    for df_idx in range(len(dfs)):
        dfs[df_idx] = divide_df_on_videos(dfs[df_idx])
    return dfs


def construct_data_loaders(video_dicts: List[Dict[str, pd.DataFrame]],
                           label_columns: List[str], feature_columns: List[str],
                           window_size: float,
                           stride: float, batch_size:int) -> List[torch.utils.data.DataLoader]:
    """ Constructs data loaders for every video_dict.

    :param video_dicts: List[Dict[str, pd.DataFrame]]
            dictionaries where keys are video names and values are pd.DataFrames
    :param label_columns: List[str]
            columns that are considered as labels (for every dataframe)
    :param feature_columns: List[str]
            columns that are considered as features (for every dataframe)
    :param window_size: float
            the length of the window in seconds
    :param stride:  float
            the length of the window stride in seconds
    :param batch_size: int
            batch size
    :return: List[torch.utils.data.DataLoader]
            list of data loaders
    """
    # here we need to construct data loaders for every video_dict.
    # every video_dict is a dictionary where keys are video names and values are pd.DataFrames
    # every pd.DataFrame has the following columns: video_name, frame_num, q1, q2, q3, emb_0, emb_1, ..., emb_255

    # !!! we consider first video_dicts as the training dataloader
    train_data_loader = video_dicts.pop(0)
    train_data_loader = TemporalEmbeddingsLoader(embeddings_with_labels=train_data_loader, label_columns=label_columns,
                                                    feature_columns=feature_columns,
                                                    window_size=window_size, stride=stride,
                                                    consider_timestamps=False,
                                                    preprocessing_functions=None, shuffle=True)
    train_data_loader = torch.utils.data.DataLoader(train_data_loader, batch_size=batch_size, shuffle=True)

    other_data_loaders = []
    for video_dict in video_dicts:
        loader = TemporalEmbeddingsLoader(embeddings_with_labels=video_dict, label_columns=label_columns,
                                            feature_columns=feature_columns,
                                          window_size=window_size, stride=stride,
                                          consider_timestamps=False,
                                          preprocessing_functions=None, shuffle=False)
        loader = torch.utils.data.DataLoader(loader, batch_size=batch_size, shuffle=False)
        other_data_loaders.append(loader)

    return [train_data_loader] + other_data_loaders

