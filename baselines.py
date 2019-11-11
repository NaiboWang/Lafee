# -*- encoding: utf-8 -*-
from datetime import datetime

import numpy as np

from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
import Lafee_3
from sklearn.utils import shuffle
import pandas as pd
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
action_list = []

def test_model(model, data, target, f_name, save=False):
    save_path = 'detailed_data/'
    pre = model.predict(data)
    # pre返回的是根据model预测data的到的标签值，所以下面的result用来计算正确率
    # result = score(pre, target)
    result = pre == target

    test_predict = scaler.inverse_transform(np.array(pre).reshape(-1, 1))  # 将归一化的数据还原成原来的数据
    test_y = scaler.inverse_transform(np.array(target).reshape(-1, 1))
    test_predict = test_predict[action_list]
    test_y = test_y[action_list]
    # 转换成留存率
    preserve_one_day_predict = np.reshape(test_predict > 24 * 3600, [-1]) + 0  # +0的含义是把True和False编程1和0
    preserve_one_day_label = np.reshape(test_y > 24 * 3600, [-1]) + 0
    preserve_three_day_predict = np.reshape(test_predict > 24 * 3600, [-1]) + 0
    preserve_three_day_label = np.reshape(test_y > 3 * 24 * 3600, [-1]) + 0
    preserve_seven_day_predict = np.reshape(test_predict > 24 * 3600, [-1]) + 0
    preserve_seven_day_label = np.reshape(test_y > 7 * 24 * 3600, [-1]) + 0
    result_one = (preserve_one_day_predict == preserve_one_day_label) + 0
    result_three = (preserve_three_day_predict == preserve_three_day_label) + 0
    result_seven = (preserve_seven_day_predict == preserve_seven_day_label) + 0
    precision_one = float(result_one.sum()/result_one.shape[0])  # 求准确率
    precision_three = float(result_three.sum()/result_three.shape[0] )  # 求准确率
    precision_seven = float(result_seven.sum()/result_seven.shape[0])  # 求准确率
    rmse = np.sqrt(mean_squared_error(test_predict, test_y))
    mae = mean_absolute_error(test_predict, test_y)
    print(f_name + "_precision:", precision_one,precision_three,precision_seven)
    # print(f_name + "_precision_three:", precision_three)
    # print(f_name + "_precision_seven:", precision_seven)
    # print(f_name + "_rmse:", rmse)
    # print(f_name + "_mae:", mae)
    if save:
        np.save(save_path + f_name + '_predict', pre)
        np.save(save_path + f_name + '_score', result)
    return np.mean(result)  # 对result矩阵求均值，即所有元素之和除以元素个数


def ann(features, actions, test_data, user_id, test_label, save=False):
    # train_size = 0.7
    model = MLPRegressor(hidden_layer_sizes=(200, 150), max_iter=1000)  # 隐藏层神经元数量为200,150
    model.fit(features, actions)  # fit之后即为训练好的model
    print("ann_loss:", model.loss_)
    result = test_model(model, test_data, test_label, user_id + '_ann')
    return result


def random_forest(features, actions, test_data, user_id, test_label, save=False):
    # train_size = 0.7
    model = RandomForestRegressor(n_estimators=500)
    # n_estimators=10：决策树的个数，越多越好，但是性能就会越差，至少100左右（具体数字忘记从哪里来的了） Random Forest（sklearn参数详解) - CSDN博客
    # http://blog.csdn.net/u012102306/article/details/52228516
    model.fit(features, actions)
    result = test_model(model, test_data, test_label, user_id + '_rf')
    return result


def svm(features, actions, test_data, user_id, test_label, save=False):
    # train_size = 0.7
    model = SVR()
    model.fit(features, actions)
    result = test_model(model, test_data, test_label, user_id + '_SVM')
    return result


def boost(features, actions, test_data, user_id, test_label, save=False):
    # train_size = 0.7
    model = AdaBoostRegressor()
    model.fit(features, actions)
    print("boost_loss:", model.loss_)
    result = test_model(model, test_data, test_label, user_id + '_rf')
    return result


def gdbt(features, actions, test_data, user_id, test_label, save=False):
    # train_size = 0.7
    model = GradientBoostingRegressor()
    model.fit(features, actions)
    print("gdbt_loss:", model.loss_)
    result = test_model(model, test_data, test_label, user_id + '_gdbt')
    return result


def DNN(features, actions, test_data, user_id, test_label, save=False):
    print("dnn_start")
    model = MLPRegressor(hidden_layer_sizes=(600, 300, 100), learning_rate_init=0.1)
    model.fit(features, actions)
    print("dnn_loss:", model.loss_)
    result = test_model(model, test_data, test_label, user_id + '_dnn')
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
    path = "origin_datas/all_origin/data.csv"
    # path = "origin_datas/all_satisfaction_aspiration/lf.csv"
    f = open(path)
    print(path)
    df = pd.read_csv(f, header=None)  # 首行也算！！！！
    df = shuffle(df)
    data = df.iloc[:, :].values
    DIM = data[0, :-1].shape[0]
    scaler_for_x = MinMaxScaler(feature_range=(0, 1))  # 最小最大值标准化，将数据转换成百分比
    scaler = MinMaxScaler(feature_range=(0, 1))
    x = np.reshape(data[:,:-1], [-1, DIM])
    y = np.reshape(data[:,-1:], [-1, DIM_TIME])
    x = scaler_for_x.fit_transform(x)  # X
    y = scaler.fit_transform(y)  # Y
    train_end = int(len(data) * 0.7)  # 分割数据
    train_x = x[:train_end, :]
    test_x = x[train_end:, :]
    train_y = y[:train_end, :]
    test_y = y[train_end:, :]

    action = data[train_end:, DIM_STATE:]
    for tt in range(action.shape[0]):
        for ttt in range(action.shape[1]):
            if action[tt][ttt] == 1:  # find the action executed this time
                action_list.append(ttt)  # save the actions with their positions in order
                break

    action_list = np.asarray(action_list)
    action_list = np.where(action_list==1)[0]
    # random_forest(train_x, train_y, test_x, TIMESTAMP, test_y)
    # svm(train_x, train_y, test_x, TIMESTAMP, test_y)
    # DNN(train_x, train_y, test_x, TIMESTAMP, test_y)
    gdbt(train_x, train_y, test_x, TIMESTAMP, test_y)