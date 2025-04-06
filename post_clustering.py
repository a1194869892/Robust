import os

import numpy as np
from munkres import Munkres
from sklearn import cluster
from scipy.sparse.linalg import svds
from sklearn.preprocessing import normalize
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
import scipy.io as sio

nmi = normalized_mutual_info_score
ami = adjusted_mutual_info_score
ari = adjusted_rand_score


def acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind_row, ind_col = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size


import numpy as np
from scipy.optimize import linear_sum_assignment


def acc_with_mapping_shuffled(y_true, y_pred):
    """
    Calculate clustering accuracy and return the mapping between true labels and predicted labels,
    considering the case where the order of predictions is shuffled.

    Parameters:
    y_true : numpy.array
        True labels with shape (n_samples,)
    y_pred : numpy.array
        Predicted labels with shape (n_samples,)

    Returns:
    float
        Accuracy, in the range [0, 1]
    dict
        Mapping between true labels and predicted labels
    """
    y_true = y_true.astype(np.int64)  # 将真实标签转换为int64类型
    assert y_pred.size == y_true.size  # 确保预测标签和真实标签的大小相同

    D = max(y_pred.max(), y_true.max()) + 1  # 找到预测标签和真实标签中的最大值，并加1
    w = np.zeros((D, D), dtype=np.int64)  # 创建一个D x D的矩阵，用于存储匹配结果

    for true_label, pred_label in zip(y_true, y_pred):
        w[pred_label, true_label] += 1  # 填充匹配矩阵

    # 使用匈牙利算法找到最大匹配
    ind_row, ind_col = linear_sum_assignment(w.max() - w)

    # 构建真实标签到伪标签的映射
    label_mapping = {j: i for i, j in zip(ind_row, ind_col)}

    # 计算准确率
    accuracy = sum([w[i, j] for i, j in zip(ind_row, ind_col)]) * 1.0 / y_pred.size

    return accuracy, label_mapping


def count_misclassifications(y_true, y_pred, true_label, predicted_labels):
    """
    Count the misclassifications where true_label is misclassified as any label in predicted_labels.

    Parameters:
    y_true : numpy.array
        True labels with shape (n_samples,)
    y_pred : numpy.array
        Predicted labels with shape (n_samples,)
    true_label : int
        The true label to be counted as misclassified
    predicted_labels : list
        List of labels to be considered as misclassifications

    Returns:
    int
        Number of misclassifications
    """
    count = 0
    for true, pred in zip(y_true, y_pred):
        if true == true_label and pred in predicted_labels:
            count += 1
    return count


# 使用混淆矩阵统计错判数量
def count_misclassifications_with_confusion_matrix(y_true, y_pred, true_label, predicted_labels):
    """
    Count the misclassifications where true_label is misclassified as any label in predicted_labels,
    using confusion matrix.

    Parameters:
    y_true : numpy.array
        True labels with shape (n_samples,)
    y_pred : numpy.array
        Predicted labels with shape (n_samples,)
    true_label : int
        The true label to be counted as misclassified
    predicted_labels : list
        List of labels to be considered as misclassifications

    Returns:
    int
        Number of misclassifications
    """
    confusion_matrix = np.zeros((max(y_true.max(), y_pred.max()) + 1, max(y_true.max(), y_pred.max()) + 1), dtype=int)
    for true, pred in zip(y_true, y_pred):
        confusion_matrix[true, pred] += 1
    misclassifications = 0
    for pred_label in predicted_labels:
        misclassifications += confusion_matrix[true_label, pred_label]
    return misclassifications


def best_map(L1, L2):
    # L1 should be the ground truth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    # 匈牙利分配算法
    m = Munkres()
    # 得到重新分配的标签的索引
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2


def err_rate(gt_s, s):
    return 1.0 - acc(gt_s, s)


def thrC(C, alpha):
    if alpha < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > alpha * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C

    return Cp


def post_proC(C, K, d, ro):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    n = C.shape[0]
    C = 0.5 * (C + C.T)
    # C = C - np.diag(np.diag(C)) + np.eye(n, n)  # good for coil20, bad for orl
    r = d * K + 1
    U, S, _ = svds(C, r, v0=np.ones(n))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** ro)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L)
    # 保存特征矩阵 U 为 .mat 文件
    # print(U.shape)  # 打印U的维度

    return grp, L, U


# def spectral_clustering(C, K, d, alpha, ro, save_dir, file_name):
#     C = thrC(C, alpha)
#     y, _, U = post_proC(C, K, d, ro)
#
#     # 如果保存目录不存在，则创建它
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#
#     # 构建保存路径
#     save_path = os.path.join(save_dir, file_name)
#
#     # # 保存特征矩阵 U 为 .mat 文件
#     sio.savemat(save_path, {'fea': U})
#
#     return y,  U

def spectral_clustering(C, K, d, alpha, ro):
    C = thrC(C, alpha)
    y, _, U = post_proC(C, K, d, ro)

    return y, U
