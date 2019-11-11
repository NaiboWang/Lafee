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
ITER_TIME = 200  # 迭代时间
DIM_STATE = 8  # state维度
DIM_ACTION = 19  # action维度
DIM_TIME = 1  # 时间维度
RNN_UNIT = 10  # rnn单元个数
INPUT_SIZE = DIM_ACTION + DIM_STATE
OUTPUT_SIZE = DIM_TIME
LEARNING_RATE = 0.1
TRAIN = True
TIME_STEP = 16
tf.reset_default_graph()
YS = -1
PATH['input'] = "origin_datas/winning_top20_classification/"
percent = 0.5
rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=10) # num_units根据实际情况设定

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

    scaled_x_data = data[:, :-3]  # X
    if YS != -1:
        scaled_y_data = data[:, YS:YS+1].reshape(-1)  # Y
    else:
        scaled_y_data = data[:, YS:].reshape(-1)  # Y
    scaler_for_x = MinMaxScaler(feature_range=(0, 1))  # 最小最大值标准化，将数据转换成百分比
    scaled_x_data = scaler_for_x.fit_transform(scaled_x_data)  # X
    normalized_train_data = scaled_x_data[: train_end]
    normalized_test_data = scaled_x_data[train_end:]
    y0 = np.reshape((scaled_y_data == 0)+0,[-1,1])
    y1 = np.reshape((scaled_y_data == 1)+0,[-1,1])
    scaled_y_data = np.concatenate([y0,y1],axis=1)
    label_train = scaled_y_data[:train_end]
    label_test = scaled_y_data[train_end:]
    # 以上获得归一化后的x和y的训练和测试集
    # however, the normalization is finished by getData now

    train_x, train_y = [], []
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i: i + time_step]
        y = label_train[i:i + time_step]
        train_x.append(x.tolist())
        train_y.append(y[-1].tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    # batch_index中存储了分批次存储好的训练数据的index，下同

    test_x, test_y = [], []
    for i in range(len(normalized_test_data) - time_step):
        x = normalized_test_data[i: i + time_step]
        y = label_test[i: i + time_step]
        test_x.append(x.tolist())
        test_y.append(y[-1].tolist())

    # train_x = np.asarray(train_x)
    # train_y = np.asarray(train_y)
    # test_x = np.asarray(test_x)
    # test_y = np.asarray(test_y)

    return batch_index, train_x, train_y, test_x, test_y

tf.set_random_seed(1)
np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 80
INPUT_SIZE = 27         # rnn input size / image width


# data

FILES = [i for i in os.listdir(PATH['input']) if re.match(r'.*csv', i)]
data = load_data(PATH['input'] + FILES[0])
batch_index, train_x, train_y, test_x, test_y = preprocess(data, batch_size=80, time_step=TIME_STEP, percentage=percent)


# tensorflow placeholders
tf_x = tf.placeholder(tf.float32, [None, TIME_STEP , INPUT_SIZE])       # shape(batch, 784)
tf_y = tf.placeholder(tf.int32, [None,2])                             # input y

# RNN

init_s = rnn_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)    # very first hidden state
outputs, _ = tf.nn.dynamic_rnn(
    rnn_cell,                   # cell you have chosen
    tf_x,                      # input
    initial_state=None,         # the initial hidden state # 不指定初始（隐含状态，不是cell），他会自动根据样本的BATCHSIZE调整大小，否则就按照初始状态中的BATCHSIZE进行调整
    dtype=tf.float32,           # must given if set initial_state = None
    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
)
# outputs[:, -1, :]指的是把最后一个step的所有batch的所有输出状态导出，然后连接到10个全连接神经元
output = tf.layers.dense(outputs[:, -1, :], 2)              # output based on the last output step

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(0.1).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]

sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)     # initialize var in graph
train_writer = tf.summary.FileWriter("RNNCLASS", sess.graph)
for i in range(200):
    for step in range(len(batch_index) - 1):    # training
        # print(batch_index[step],batch_index[step + 1])
        b_x = train_x[batch_index[step]: batch_index[step + 1]]
        b_y = train_y[batch_index[step]: batch_index[step + 1]]
        # t4,t5,t6 = sess.run([outputs,h_n,h_c], {tf_x: b_x, tf_y: b_y})
        _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y}) # h_n和h_c在不训练的时候也会根据数据不同而不同
        # t1, t2, t3= sess.run([outputs,h_n,h_c], {tf_x: b_x, tf_y: b_y})
    if i % 40 == 0:      # testing
        accuracy_ = sess.run(accuracy, {tf_x: test_x, tf_y: test_y})
        ou = sess.run(outputs, {tf_x: test_x, tf_y: test_y})
        # hs = sess.run(h_n, {tf_x: test_x, tf_y: test_y})
        print('train loss: %.4f' % loss_, '| test accuracy: %.4f' % accuracy_)