import math
import scipy.io as sio
import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import normalize, minmax_scale

data_names = {"human_ESC": "human_ESC", "human_brain": 'human_brain', "time_course": "time_course"}


def get_datamat(data_name, shape, device):

    filename = "./datasets/" + data_names[data_name] + ".mat"
    data = sio.loadmat(filename)

    features, labels = data['fea'], data['label']
    original_size = int(np.sqrt(features.shape[1]))
    shape = math.ceil((np.sqrt(features.shape[1])))
    if shape % 2 != 0:
        shape += 1
    print(original_size, shape)
    padded_features = np.pad(features, ((0, 0), (0, shape * shape - features.shape[1])), mode='constant')

    data['fea'] = padded_features
    print(data['fea'].shape)
    idx = np.argsort(data['fea'].std(0))[::-1][:shape * shape]
    features, labels = data['fea'][:, idx], data['label']
    ori_matrix = data['fea'][:, idx]
    features = normalize(features, norm='l1', axis=1)
    features = features.reshape((-1, 1, shape, shape))

    dropout_matrix = np.ones_like(ori_matrix)
    dropout_matrix[ori_matrix == 0] = 0
    idx = np.argwhere(np.all(dropout_matrix[..., :] == 0, axis=0))  # 列为全0的数据不具备参考性，直接取0
    dropout_matrix[:, idx] = 1
    dropout_matrix = dropout_matrix.reshape((-1, 1, shape, shape))

    features = torch.from_numpy(features).float().to(device)
    labels = np.squeeze(labels - np.min(labels))
    dropout_matrix = dropout_matrix.astype(np.float64)
    dropout_matrix = torch.from_numpy(dropout_matrix).float().to(device)
    print('before return', features.shape)
    return features, labels, dropout_matrix
