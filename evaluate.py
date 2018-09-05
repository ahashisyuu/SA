import numpy as np


def compute_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))


def binary_prf(y_true, y_pred):
    """
        y_true:真实标签的ndarray，形状为(N,), N为所有样本的个数
        y_pred:预测值的ndarray，形状一样

    """
    assert y_true.shape == y_pred.shape

    TP = y_true[y_pred == 1].sum()
    TP_FP = y_pred.sum()
    TP_FN = y_true.sum()

    # print(TP, TP_FP, TP_FN)
    precision = (TP + 1) / (TP_FP + 1)
    recall = (TP + 1) / (TP_FN + 1)
    f1_score = 2*precision*recall / (precision + recall)

    return precision, recall, f1_score


def categorical_prf(y_true, y_pred):
    """
        y_true:真实标签的ndarray，形状为(N, class_num), N为所有样本的个数， class_num为类别数，本次比赛固定为每个层次类别数为4
        y_pred:预测值的ndarray，形状一样

    """
    assert y_true.shape == y_pred.shape

    # 将y_pred和y_true变为编号形式
    y_pred_index = y_pred.argmax(axis=-1)
    class_num = y_pred_index.max() + 1
    assert class_num == y_pred.shape[1]
    y_true_index = y_true.argmax(axis=-1)

    # 将class_num的多分类分为class_num个2分类组合并计算PRF
    group_prf = []
    group_f = 0
    for i in range(class_num):
        new_y_true = (y_true_index == i).astype('int32')
        new_y_pred = (y_pred_index == i).astype('int32')
        prf = binary_prf(new_y_true, new_y_pred)
        group_prf.append(prf)
        group_f += prf[2]

    group_f /= class_num

    acc = np.mean((y_pred_index == y_true_index).astype('int32'))

    return group_prf, group_f, acc




