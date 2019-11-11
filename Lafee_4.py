# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops import math_ops
from c_rnn_cell_impl import c_BasicLSTMCell, c_SainRNNCell, c_BasicRNNCell, c_LaFeeCell
import numpy as np
import pandas as pd
import os
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import matplotlib.pyplot as plt
import warnings
import csv

np.set_printoptions(threshold='nan')  # 全部输出
warnings.filterwarnings('ignore')

# Global variables
PATH = {'raw': '../../fb_dataset/fb_player_event_json/dataset/',
        'input': '../../fb_dataset/fb_player_event_json/dataset_input/',
        'washed_json': '../../fb_dataset/fb_player_event_json/dataset_washed_json/'}
# 文件夹路径
FILES = [i for i in os.listdir(PATH['input']) if re.match(r'.*csv', i)]
# 读取文件列表
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())  # 当前时间戳
ITER_TIME = 3000  # 迭代时间
DIM_STATE = 8  # state维度
DIM_ACTION = 19  # action维度
DIM_TIME = 1  # 时间维度
RNN_UNIT = 10  # rnn单元个数
INPUT_SIZE = DIM_ACTION + DIM_STATE
OUTPUT_SIZE = DIM_TIME
LEARNING_RATE = 0.1
TRAIN = True
TIME_STEP = 15
tf.reset_default_graph()


def print_important(str):  # 以粉红色打印出信息
    print('\033[1;35m', str, '\033[0m')


# Read data from files and divide the dimensions
def load_data(file_path, dim_list=[DIM_STATE + DIM_ACTION, DIM_TIME], split=False):
    f = open(file_path)
    print("file_path:", file_path)
    df = pd.read_csv(f, header=None)  # 首行也算！！！！
    if split:  # 如果要把state与action分开
        point = 0
        data = []
        for dim in dim_list:
            data.append(df.iloc[:, point:(point + dim)].values)
            point += dim
        return data
    else:
        return df.iloc[:, :].values


def preprocess(data, batch_size=80, time_step=TIME_STEP, percentage=0.7):
    print_important('>>>> preprocessing...')
    train_end = int(len(data) * percentage)  # 分割数据
    batch_index = []

    scaled_x_data = data[:, :-1]  # X
    scaled_y_data = data[:, -1:].reshape(-1)  # Y

    normalized_train_data = scaled_x_data[: train_end]
    normalized_test_data = scaled_x_data[train_end:]
    label_train = scaled_y_data[:train_end]
    label_test = scaled_y_data[train_end:]
    # 以上获得归一化后的x和y的训练和测试集
    # however, the normalization is finished by getData now

    train_x, train_y = [], []
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i: i + time_step]
        y = label_train[i:i + time_step, np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    # batch_index中存储了分批次存储好的训练数据的index，下同

    test_x, test_y = [], []
    for i in range(len(normalized_test_data) - time_step):
        x = normalized_test_data[i: i + time_step]
        y = label_test[i: i + time_step, np.newaxis]
        test_x.append(x.tolist())
        test_y.append(y.tolist())

    # train_x = np.asarray(train_x)
    # train_y = np.asarray(train_y)
    # test_x = np.asarray(test_x)
    # test_y = np.asarray(test_y)

    return batch_index, train_x, train_y, test_x, test_y


# 得到多个文件的数据
def getData(num=10, batch_size=80, time_step=TIME_STEP, percentage=0.7,mode = 0):
    batch_index = []
    train_x = []
    train_y = []
    test_x = []
    test_y = []
    action_list = []
    is_logout_train = []
    is_logout_test = []
    user_index = [0]
    user_id = [0]
    percent = percentage
    for i in range(num):
        # 100140101
        data = load_data(PATH['input'] + FILES[i])
        # data = load_data(PATH['input'] + "100140101.csv")
        if mode == 1: # 如果是1，一部分用户的全部用来训练、一部分用户全部用来测试的版本
            if i >= num*percentage:
                percent = 0.0
            else:
                percent = 1.0
        batch_index_t, train_x_t, train_y_t, test_x_t, test_y_t = preprocess(data, batch_size, time_step,
                                                                             percentage=percent)
        # 先从每一个文件得到原始数据，然后放入总数组中
        if len(test_x_t) > 0:
            test_x.extend(test_x_t)
            test_y.extend(test_y_t)
            user_index.append(user_index[-1] + len(test_y_t))  # record the index of each test batch
            user_id.append(FILES[i][:-4])
        if len(train_x_t) > 0:
            train_x.extend(train_x_t)
            train_y.extend(train_y_t)
            # 下面正确存储了
            if len(batch_index) > 0:
                for j in range(len(batch_index_t)):
                    batch_index_t[j] += batch_index[-1]  # adjust the batch index with offset
            batch_index.extend(batch_index_t)

    t_a_l = np.asarray(test_x)[:, :, DIM_STATE:]
    t_a_t = np.asarray(train_x)[:, :, DIM_STATE:]
    for t in range(t_a_l.shape[0]):
        for tt in range(t_a_l.shape[1]):
            for ttt in range(t_a_l.shape[2]):
                if t_a_l[t][tt][ttt] == 1:  # find the action executed this time
                    action_list.append(ttt)  # save the actions with their positions in order
                    if ttt == 1:  # judge whether the action is logout (index of logout is 1)
                        is_logout_test.append(True)
                    else:
                        is_logout_test.append(False)
                    break
    for t in range(t_a_t.shape[0]):
        for tt in range(t_a_t.shape[1]):
            for ttt in range(t_a_t.shape[2]):
                if t_a_t[t][tt][ttt] == 1:
                    if ttt == 1:
                        is_logout_train.append(True)
                    else:
                        is_logout_train.append(False)
                    break
    action_list = np.asarray(action_list).reshape([-1, time_step])
    len_train_x = len(train_x)
    train_x.extend(test_x)
    train_y.extend(test_y)

    train_x = np.asarray(train_x).reshape([-1, DIM_STATE + DIM_ACTION])
    train_y = np.asarray(train_y).reshape([-1, DIM_TIME])

    scaler_for_x = MinMaxScaler(feature_range=(0, 1))  # 最小最大值标准化，将数据转换成百分比
    scaler_for_y = MinMaxScaler(feature_range=(0, 1))
    scaled_x_data = scaler_for_x.fit_transform(train_x)  # X
    scaled_y_data = scaler_for_y.fit_transform(train_y).reshape(-1)  # Y

    scaled_x_data = scaled_x_data.reshape([-1, time_step, DIM_STATE + DIM_ACTION])
    scaled_y_data = scaled_y_data.reshape([-1, time_step, DIM_TIME])

    train_x_final = scaled_x_data[:len_train_x]
    train_y_final = scaled_y_data[:len_train_x]
    test_x_final = scaled_x_data[len_train_x:]
    test_y_final = scaled_y_data[len_train_x:]

    print("训练和测试数据长度为：", train_x_final.shape[0], test_x_final.shape[0])
    return batch_index, train_x_final, train_y_final, test_x_final, test_y_final, scaler_for_y, action_list, test_y, user_index, user_id, is_logout_test, is_logout_train



def lafee_model(X, ACTION):
    print_important('>>>> using lafee_model now...')
    batch_size = tf.shape(X)[0]

    X_state = X[:, :, :DIM_STATE]
    X_action = X[:, :, DIM_STATE:]

    with tf.variable_scope('rnn_satisfaction'):
        satis_cell = c_LaFeeCell(DIM_STATE, activation=math_ops.sigmoid, name='satis_cell')
        # satis_cell = c_BasicLSTMCell(DIM_STATE, activation=math_ops.sigmoid, name='satis_cell')
        init_state = satis_cell.zero_state(batch_size, dtype=tf.float32)
        rnn_satis_, satisfaction_in = tf.nn.dynamic_rnn(satis_cell, X_state, initial_state=init_state, dtype=tf.float32)

    with tf.variable_scope('rnn_aspiration'):
        aspir_cell = c_LaFeeCell(DIM_ACTION, activation=math_ops.sigmoid, name='aspir_cell')
        # aspir_cell = c_BasicLSTMCell(DIM_ACTION, activation=math_ops.sigmoid, name='aspir_cell')
        init_state2 = aspir_cell.zero_state(batch_size, dtype=tf.float32)
        rnn_aspir_, aspiration_out = tf.nn.dynamic_rnn(aspir_cell, X_action, initial_state=init_state2, dtype=tf.float32)

    rnn_satis = tf.reshape(rnn_satis_, [-1, DIM_STATE])
    X_action_new = tf.reshape(X_action, [-1, DIM_ACTION])
    action_sat = tf.concat([rnn_satis, X_action_new], 1)  # 拼接action和satisfaction
    tin_asp_input = tf.layers.dense(action_sat, DIM_ACTION, name='tin_asp_input')
    tin_asp_input = tf.reshape(tin_asp_input,[-1,TIME_STEP,DIM_ACTION]) # 转换成RNN需要格式
    sat_asp, aspiration_in = tf.nn.dynamic_rnn(aspir_cell, tin_asp_input, initial_state=init_state2, dtype=tf.float32)
    sat_asp = tf.reshape(sat_asp,[-1,DIM_ACTION])
    tin = tf.layers.dense(sat_asp, 1, name='tin')  # output layer 输出层增加方式

    rnn_aspir = tf.reshape(rnn_aspir_, [-1, DIM_ACTION])
    X_state_new = tf.reshape(X_state, [-1, DIM_STATE])
    state_asp = tf.concat([rnn_aspir, X_state_new], 1)  # 拼接state和aspiration
    tout_sat_input = tf.layers.dense(state_asp, DIM_STATE, name='tout_sat_input')
    tout_sat_input = tf.reshape(tout_sat_input,[-1,TIME_STEP,DIM_STATE]) # 转换成RNN需要格式
    asp_sat, satisfaction_out = tf.nn.dynamic_rnn(satis_cell, tout_sat_input, initial_state=init_state,dtype=tf.float32)  # 再过LFs模型
    asp_sat = tf.reshape(asp_sat,[-1,DIM_STATE])
    tout = tf.layers.dense(asp_sat, 1, name='tout')  # output layer 输出层增加方式

    ACTION_ = tf.reshape(ACTION, [-1])
    pred = tf.where(ACTION_, tout, tin)
    ACTION__ = tf.reshape(ACTION[:,-1,:],[-1]) # satisfaction和aspiration都是TIMESTEP中的最后一步，因此只取最后一步
    satisfaction = tf.where(ACTION__, satisfaction_out, satisfaction_in)  # satisfaction的选择
    aspiration = tf.where(ACTION__, aspiration_out, aspiration_in)  # aspiration的选择
    return pred, satisfaction, aspiration

# 首词大写用于pycharm调试
def TEST_model(sess, pred, satisfaction, aspiration, t_X, t_Y, pl_X, scaler, actionlist, ACTION):
    print_important('>>>> testing...')
    test_predict = []
    test_output = []
    test_state = []

    for step in range(len(t_X)):
        prob, output_, state_ = sess.run([pred, satisfaction, aspiration],
                                         feed_dict={pl_X: [t_X[step]], ACTION: [actionlist[step]]})
        predict = prob.reshape((-1))
        test_predict.extend(predict)
        test_output.extend(output_)
        test_state.extend(state_)

    test_predict = scaler.inverse_transform(np.array(test_predict).reshape(-1, 1))  # 将归一化的数据还原成原来的数据
    test_y = scaler.inverse_transform(np.array(t_Y).reshape(-1, 1))
    mae = mean_absolute_error(y_pred=test_predict, y_true=test_y)
    rmse = np.sqrt(mean_squared_error(test_predict, test_y))
    test_y = np.reshape(np.reshape(test_y, [-1, TIME_STEP, 1])[:, -1, :], [-1, 1])  # 只保留15步最后一步的预测值
    test_predict = np.reshape(np.reshape(test_predict, [-1, TIME_STEP, 1])[:, -1, :], [-1, 1])  # 只保留15步最后一步的预测值
    print('mae:', mae, '    rmse:', rmse)
    return test_predict, test_output, test_state, mae, rmse, test_y


def train(batch_size=80, time_step=TIME_STEP, percentage=0.7, model=lafee_model, cell=c_BasicRNNCell,
          summary_dir=TIMESTAMP, num=1, mode=0):
    print_important('>>>> training...')

    # 训练数据格式：15X15X27 15X15X1
    # 测试数据格式：1X15X27 1X15X1
    X = tf.placeholder(tf.float32, shape=[None, time_step, INPUT_SIZE])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, OUTPUT_SIZE])
    ACTION = tf.placeholder(tf.bool, shape=[None, time_step, 1])  # 判断行为是哪个
    batch_index, train_x, train_y, test_x, test_y, scaler_for_y, action_list, result_list, user_index, user_id, is_logout_test, is_logout_train = getData(
        num=num, mode=mode)  # 从库中选择num个文件合并训练
    is_logout_test = np.asarray(is_logout_test).reshape([-1, time_step, 1])
    is_logout_train = np.asarray(is_logout_train).reshape([-1, time_step, 1])
    pred, satisfaction, aspiration = model(X, ACTION)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))  # 求预测值与标签值的均方误差
    loss_summary = tf.summary.scalar('loss', loss)

    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    with tf.Session() as sess:

        train_log_dir = 'logs/train/' + summary_dir
        test_log_dir = 'logs/test/' + summary_dir
        try:
            os.mkdir('origin_logs')
        except:
            pass
        os.mkdir('origin_logs/' + summary_dir)

        # write the real sequence of action and time into files
        f1 = open("origin_logs/" + summary_dir + "action.csv", 'w', newline='')
        f5 = open("origin_logs/" + summary_dir + "time.csv", 'w', newline='')
        csv_write = csv.writer(f1)
        for data in action_list.tolist():
            csv_write.writerow(data)
        csv_write = csv.writer(f5)
        for data in np.asarray(result_list).reshape([-1, time_step]).tolist():
            csv_write.writerow(data)
        f1.close()
        f5.close()

        f2 = open("origin_logs/" + summary_dir + "satisfaction.csv", 'w', newline='')
        f3 = open("origin_logs/" + summary_dir + "aspiration.csv", 'w', newline='')
        f4 = open("origin_logs/" + summary_dir + "error.csv", 'w', newline='')
        f6 = open("origin_logs/" + summary_dir + "loss.csv", 'w', newline='')
        f7 = open("origin_logs/" + summary_dir + "user_index.csv", 'w', newline='')
        f8 = open("origin_logs/" + summary_dir + "pred.csv", 'w', newline='')
        f9 = open("origin_logs/" + summary_dir + "user_id.csv", 'w', newline='')
        csv_write2 = csv.writer(f2)
        csv_write3 = csv.writer(f3)
        csv_write4 = csv.writer(f4)
        csv_write6 = csv.writer(f6)
        csv_write7 = csv.writer(f7)
        csv_write8 = csv.writer(f8)
        csv_write9 = csv.writer(f9)
        csv_write7.writerow(user_index)  # 写入index
        csv_write9.writerow(user_id)  # save user id
        f7.close()
        f9.close()
        merged = tf.summary.merge([loss_summary])
        train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        test_writer = tf.summary.FileWriter(test_log_dir)

        sess.run(tf.global_variables_initializer())
        # 并没有对batch_index进行打乱
        for i in range(ITER_TIME):
            loss_ = None
            summary_train = None
            for step in range(len(batch_index) - 1):
                if batch_index[step] != batch_index[step + 1]:  # 注意这行，如果两个值相同意味着来了新用户

                    summary_train, _, loss_, pred_ = sess.run(
                        [merged, train_op, loss, pred],
                        feed_dict={X: train_x[batch_index[step]: batch_index[step + 1]],
                                   Y: train_y[batch_index[step]: batch_index[step + 1]],
                                   ACTION: is_logout_train[batch_index[step]: batch_index[step + 1]]})
                    csv_write6.writerow([loss_])
            if i % 100 == 0:
                print('iter:', i, 'loss:', loss_)
                test_predict, test_output, test_state, mae, rmse, test_yy = TEST_model(sess, pred, satisfaction,
                                                                                       aspiration, test_x, test_y, X,
                                                                                       scaler_for_y, is_logout_test,
                                                                                       ACTION)
                # print("test_predict:", test_predict)
                csv_write4.writerow([mae, rmse])  # 记录error
                pred_y = np.concatenate([test_predict, test_yy], axis=1)
                for data in np.asarray(pred_y).tolist():
                    csv_write8.writerow(data)  # predition and ground truth
                # print("test_state(aspiration):", test_state)

            train_writer.add_summary(summary_train, i)

        for data in np.asarray(test_output).tolist():
            csv_write2.writerow(data)  # satisfaction
        for data in np.asarray(test_state).tolist():
            csv_write3.writerow(data)  # aspiration
        f2.close()
        f3.close()
        f4.close()
        f6.close()
        f7.close()
        f8.close()

    train_writer.close()
    test_writer.close()
    return test_predict, test_output, test_state


if __name__ == '__main__':
    # getData(10)
    # data = load_data(PATH['input'] + FILES[0])
    summary_dir = TIMESTAMP + '_lafee_model/'
    test_predict, test_output, test_state = train(model=lafee_model, summary_dir=summary_dir, num=20, mode = 1)  # num为用户读入用户数目
"""
pass
"""
