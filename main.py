import os
import warnings
import random
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
DATA_PATH = './datasets/'
estimator_list = [0, 1, 2, 3, 4]
warnings.filterwarnings("ignore")


def select_estimator(num):
    if num == 0:
        estimator = SVC()
    elif num == 1:
        estimator = KNeighborsClassifier()
    elif num == 2:
        estimator = DecisionTreeClassifier()
    elif num == 3:
        estimator = GaussianNB()
    elif num == 4:
        estimator = LogisticRegression()
    return estimator


def accuracy(estimator, features, label_set):
    kf = KFold(n_splits=10)
    m_acc = []
    for trainIdx, testIdx in kf.split(features):
        train_set = features[trainIdx]
        test_set = features[testIdx]
        train_label = label_set[trainIdx]
        test_label = label_set[testIdx]
        estimator.fit(train_set, train_label)
        m_acc.append(estimator.score(test_set, test_label))

    return np.mean(m_acc)


def s_ifs(feature_set, label_set, feature_idx, k, d):
    col_ = feature_set.shape[1]
    best_fs = set([])
    decrease_time = 0
    cur_acc = 0.0
    for idx in range(col_ - k):
        m_acc = 0.0
        locs = [x for x in range(k, k + idx + 1)]  # 初始化连续特征索引数组，每次多加入一个后继特征
        X = np.array(feature_set)[:, list(np.array(feature_idx)[locs])]  # 根据索引数组构造数据子集
        for i in estimator_list:  # 执行多种分类算法中选出最大准确率
            acc = accuracy(select_estimator(i), X, label_set)
            if acc > m_acc:
                m_acc = acc
        if m_acc > cur_acc:  # 当前连续特征子集准确率比上一个子集效果好
            decrease_time = 0  # 重新计数
            best_fs = set(locs)
        else:
            decrease_time += 1  # 当前特征子集效能下降
            if decrease_time == d + 1:  # 下降次数超过阈值
                break
        cur_acc = m_acc  # cur_acc为上一子集的准确率
    print(len(best_fs))
    return best_fs


# RIFS算法默认d=4, 随机起始位置个数为特征数的一半
def r_ifs(feature_set, label_set, feature_idx, d):
    solution = []
    row_, col_ = feature_set.shape
    k_ = int(col_ * 0.5)-1  # 起始位置个数
    random_loc = random.sample(list(range(1, col_)), k_)  # 获取随机起始位置
    random_loc.append(0)  # 确保0在里面
    for loc in random_loc:  # 从每个随机位置开始选出一个连续特征子集
        solution.append(s_ifs(feature_set, label_set, feature_idx, loc, d))

    best_fs = solution[0]
    cur_acc = 0.0
    for idx in range(0, k_):  # 从得到的多个连续特征子集中选出效能最好的子集
        m_acc = 0.0
        for i in estimator_list:
            acc = accuracy(select_estimator(i),
                           np.array(feature_set)[:,
                           solution[idx]],
                           label_set)
            if acc > m_acc:
                m_acc = acc
        if m_acc > cur_acc:
            cur_acc = m_acc
            best_fs = solution[idx]

    return best_fs


for data_name in os.listdir(DATA_PATH):
    data_array = pd.read_table(os.path.join(DATA_PATH, data_name),
                               header=None,
                               low_memory=False,
                               index_col=0).transpose().to_numpy()
    features = data_array[:, 1:].astype('float')
    labels = data_array[:, 0]
    for number, label in enumerate(list(set(labels))):
        labels[np.where(labels == label)] = number  # 标签数字化
    labels = labels.astype('int')
    row_num, col_num = np.array(features).shape
    t_test_index = [x for x in range(col_num)]
    t_test_value = [0] * col_num
    for idx in range(col_num):  # t-test计算p值
        t_test_value[idx] = stats.ttest_ind(features[:, idx], labels)[1]
    t_test_index.sort(key=lambda x: t_test_value[x], reverse=True)  # 按p值递减顺序构造索引数组
    res_fs = r_ifs(features, labels, t_test_index, 4)
    print(np.array(res_fs).shape)
