# -*- encoding: utf-8 -*-
from datetime import datetime

import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.externals import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import Lafee_3

""" 
系统的介绍下sklearn的用法 ML神器：sklearn的快速使用 - ML小菜鸟 - 博客园
https://www.cnblogs.com/lianyingteng/p/7811126.html
# 拟合模型 训练数据
model.fit(X_train, y_train)
# 模型预测 预测数据
model.predict(X_test)

# 获得这个模型的参数
model.get_params()
# 为模型进行打分
model.score(data_X, data_y) # 线性回归：R square； 分类问题： acc

"""
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())  # 当前时间戳
DIM_STATE = 8  # state维度
DIM_ACTION = 19  # action维度
DIM_TIME = 1  # 时间维度
scaler = None


def TEST_model(model, data, target, f_name, save=False):
    save_path = 'detailed_data/'
    pre = np.reshape(model.predict(data),[-1,1])
    # pre返回的是根据model预测data的到的标签值，所以下面的result用来计算正确率
    # result = score(pre, target)
    result = (pre == target)+0
    precision = float(result.sum()/result.shape[0])  # 求准确率
    print("precision_"+f_name+":",precision)
    if save:
        np.save(save_path + f_name + '_predict', pre)
        np.save(save_path + f_name + '_score', result)
    return np.mean(result)  # 对result矩阵求均值，即所有元素之和除以元素个数


def ann(features, actions, test_data, user_id, test_label, save=False):
    # train_size = 0.7
    model = MLPClassifier(hidden_layer_sizes=(200, 150), max_iter=1000)  # 隐藏层神经元数量为200,150
    model.fit(features, actions)  # fit之后即为训练好的model
    result = TEST_model(model, test_data, test_label, user_id + '_ann')
    return result


def random_forest(features, actions, test_data, user_id, test_label, save=False):
    # train_size = 0.7
    model = RandomForestClassifier(n_estimators=500)
    # n_estimators=10：决策树的个数，越多越好，但是性能就会越差，至少100左右（具体数字忘记从哪里来的了） Random Forest（sklearn参数详解) - CSDN博客
    # http://blog.csdn.net/u012102306/article/details/52228516
    model.fit(features, actions)
    result = TEST_model(model, test_data, test_label, user_id + '_rf')
    return result


def svm(features, actions, test_data, user_id, test_label, save=False):
    # train_size = 0.7
    model = SVC()
    model.fit(features, actions)
    result = TEST_model(model, test_data, test_label, user_id + '_SVM')
    return result


def boost(features, actions, test_data, user_id, test_label, save=False):
    # train_size = 0.7
    model = AdaBoostClassifier()
    model.fit(features, actions)
    result = TEST_model(model, test_data, test_label, user_id + '_rf')
    return result


def gdbt(features, actions, test_data, user_id, test_label, save=False):
    # train_size = 0.7
    model = GradientBoostingClassifier()
    model.fit(features, actions)
    result = TEST_model(model, test_data, test_label, user_id + '_gdbt')
    return result


def DNN(features, actions, test_data, user_id, test_label, save=False):
    model = MLPClassifier(hidden_layer_sizes=(600, 300, 100), learning_rate_init=0.1)
    model.fit(features, actions)
    result = TEST_model(model, test_data, test_label, user_id + '_dnn')
    return result


# def smooth(x):
#     x /= 10
#     if abs(x) > 1:
#         return x
#     else:
#         return pow(x, 3)


# def sigmoid(x):
#     return (1. / (1 + np.exp(-x)) - 0.5) * 40


if __name__ == '__main__':
    path = "origin_datas/battle_bottom20_classification/classfication.csv"
    f = open(path)
    print(path)
    df = pd.read_csv(f, header=None)  # 首行也算！！！！
    data = df.iloc[:, :].values
    x = data[:, :-3]
    scaler = MinMaxScaler(feature_range=(0, 1))
    x = scaler.fit_transform(x)  # X
    train_end = int(len(data) * 0.4)  # 分割数据
    train_x = np.reshape(x[:train_end, :], [-1, DIM_STATE + DIM_ACTION])  # X
    test_x = np.reshape(x[train_end:, :], [-1, DIM_STATE + DIM_ACTION])  # X
    train_y1 = np.reshape(data[:train_end, -1:], [-1, DIM_TIME])
    test_y1 = np.reshape(data[train_end:, -1:], [-1, DIM_TIME])
    train_y3 = np.reshape(data[:train_end, -2:-1], [-1, DIM_TIME])
    test_y3 = np.reshape(data[train_end:, -2:-1], [-1, DIM_TIME])
    train_y7 = np.reshape(data[:train_end, -3:-2], [-1, DIM_TIME])
    test_y7 = np.reshape(data[train_end:, -3:-2], [-1, DIM_TIME])
    print("one day:")
    random_forest(train_x, train_y1, test_x, TIMESTAMP, test_y1)
    svm(train_x, train_y1, test_x, TIMESTAMP, test_y1)
    DNN(train_x, train_y1, test_x, TIMESTAMP, test_y1)
    gdbt(train_x, train_y1, test_x, TIMESTAMP, test_y1)
    print("three day:")
    random_forest(train_x, train_y3, test_x, TIMESTAMP, test_y3)
    svm(train_x, train_y3, test_x, TIMESTAMP, test_y3)
    DNN(train_x, train_y3, test_x, TIMESTAMP, test_y3)
    gdbt(train_x, train_y3, test_x, TIMESTAMP, test_y3)
    print("seven day:")
    random_forest(train_x, train_y7, test_x, TIMESTAMP, test_y7)
    svm(train_x, train_y7, test_x, TIMESTAMP, test_y7)
    DNN(train_x, train_y7, test_x, TIMESTAMP, test_y7)
    gdbt(train_x, train_y7, test_x, TIMESTAMP, test_y7)