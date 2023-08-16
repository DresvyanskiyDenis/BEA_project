from typing import List, Dict, Callable, Optional

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


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
        dfs[df_idx] = dfs[df_idx][new_order]

    # the data format of the df is the following:
    # video_name, frame_num, q1, q2, q3, emb_0, emb_1, ..., emb_255
    # we first need to divide the data on videos so that every video will be represented as Dict[str, pd.DataFrame]
    # DataFrame has the same columns: video_name, frame_num, q1, q2, q3, emb_0, emb_1, ..., emb_255
    # str is the name of the video
    for df_idx in range(len(dfs)):
        dfs[df_idx] = divide_df_on_videos(dfs[df_idx])
    return dfs


def compute_class_weights(data:Dict[str, pd.DataFrame])->List[np.array]:
    # concat all dataframes
    df = pd.concat(list(data.values()))
    # get number of samples for every class
    q1 = df['q1'].value_counts().sort_index().values
    q2 = df['q2'].value_counts().sort_index().values
    q3 = df['q3'].value_counts().sort_index().values
    # compute class weights
    class_weights = [1./(q1/q1.sum()), 1./(q2/q2.sum()), 1./(q3/q3.sum())]
    # normalize so that sum of weights is 1
    class_weights = [w/w.sum() for w in class_weights]
    return class_weights

class DataLoader(Dataset):

    def __init__(self, embeddings_with_labels:Dict[str, pd.DataFrame], label_columns:List[str], feature_columns:List[str],
                 preprocessing_functions:List[Callable]=None, padding:Optional[bool]=False, padding_mode:Optional[str]='zeros'):
        self.embeddings_with_labels = embeddings_with_labels
        self.label_columns = label_columns
        self.feature_columns = feature_columns
        self.preprocessing_functions = preprocessing_functions
        self.padding = padding
        self.padding_mode = padding_mode

        # split features and labels from the embeddings_with_labels
        self.features = {}
        self.labels = {}
        for key, df in self.embeddings_with_labels.items():
            self.features[key] = df[self.feature_columns]
            self.labels[key] = df[self.label_columns]

        # transform labels taking the mode of every column
        for key, df in self.labels.items():
            self.labels[key] = df.mode(axis=0).values.squeeze()

        # pad features to the maximum length if needed
        if self.padding:
            self.features = self.__pad_features(self.features, self.padding_mode)





    def __len__(self):
        return len(self.embeddings_with_labels)

    def __getitem__(self, idx):
        # get the data and labels using pointer
        key = list(self.embeddings_with_labels.keys())[idx]
        features = self.features[key].values
        labels = self.labels[key]
        # preprocess embeddings if needed
        if self.preprocessing_functions is not None:
            for preprocess_function in self.preprocessing_functions:
                features = preprocess_function(features)
        # The output shape is (seq_len, num_features)
        return features, labels

    def __pad_features(self, features, padding_mode):
        if padding_mode == 'max':
            max_length = max([df.shape[0] for df in features.values()])
            for key, df in features.items():
                if df.shape[0] < max_length:
                    padding = np.zeros((max_length - df.shape[0], df.shape[1]))
                    padding = pd.DataFrame(padding, columns=df.columns)
                    features[key] = pd.concat([df, padding], axis=0)
        elif padding_mode == 'average':
            average_length = int(np.mean([df.shape[0] for df in features.values()]))
            for key, df in features.items():
                if df.shape[0] < average_length:
                    padding = np.zeros((average_length - df.shape[0], df.shape[1]))
                    padding = pd.DataFrame(padding, columns=df.columns)
                    features[key] = pd.concat([df, padding], axis=0)
                else:
                    features[key] = df.iloc[:average_length, :]
        else:
            raise ValueError("padding_mode should be either 'max' or 'average'")
        return features



if __name__ == "__main__":
    # params
    path_to_train_df = "/nfs/home/ddresvya/scripts/BEA/extracted_embeddings/train_embeddings.csv"
    path_to_dev_df = "/nfs/home/ddresvya/scripts/BEA/extracted_embeddings/dev_embeddings.csv"
    path_to_test_df = "/nfs/home/ddresvya/scripts/BEA/extracted_embeddings/test_embeddings.csv"
    print("Start of the script....")
    # load data
    print("Loading data....")
    data = load_data(paths_to_dfs=[path_to_train_df, path_to_dev_df, path_to_test_df])

    # test DataLoader
    print("Testing DataLoader....")
    train_data_loader = DataLoader(embeddings_with_labels=data[0], label_columns=['q1', 'q2', 'q3'],
                                      feature_columns=[f'emb_{i}' for i in range(256)],
                                        preprocessing_functions=None, padding=True, padding_mode='average')
    train_data_loader = torch.utils.data.DataLoader(train_data_loader, batch_size=1, shuffle=True)

    for i, (x, y) in enumerate(train_data_loader):
        print(x.shape)
        print(y.shape)
        print('-------------------')
        if i == 10:
            break

