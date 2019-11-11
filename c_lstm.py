import tensorflow as tf
from tensorflow.python.ops import math_ops
from c_rnn_cell_impl import c_BasicLSTMCell, c_SainRNNCell, c_BasicRNNCell,c_LaFeeCell
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

np.set_printoptions(threshold='nan')  #全部输出
warnings.filterwarnings('ignore')

# Global variables
PATH = {'raw': '../../fb_dataset/fb_player_event_json/dataset/',
        'input': '../../fb_dataset/fb_player_event_json/dataset_input/',
        'washed_json': '../../fb_dataset/fb_player_event_json/dataset_washed_json/'}
# 文件夹路径
FILES = [i for i in os.listdir(PATH['input']) if re.match(r'.*csv', i)]
# 读取文件列表
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())  # 当前时间戳
ITER_TIME = 1000  # 迭代时间
DIM_STATE = 8  # state维度
DIM_ACTION = 19  # action维度
DIM_TIME = 1  # 时间维度
RNN_UNIT = 10  # rnn单元个数
INPUT_SIZE = DIM_ACTION + DIM_STATE
OUTPUT_SIZE = DIM_TIME
LEARNING_RATE = 0.1
TRAIN = True

tf.reset_default_graph()


def print_important(str):  # 以粉红色打印出信息
    print('\033[1;35m', str, '\033[0m')


# Read data from files and divide the dimensions
def load_data(file_path, dim_list, split=False):
    f = open(file_path)
    print("file_path:",file_path)
    df = pd.read_csv(f)
    if split:  # 如果要把state与action分开
        point = 0
        data = []
        for dim in dim_list:
            data.append(df.iloc[:, point:(point + dim)].values)
            point += dim
        return data
    else:
        return df.iloc[:, :].values


def preprocess(data, batch_size=80, time_step=15, percentage=0.7):
    print_important('>>>> preprocessing...')
    train_end = int(len(data) * percentage)  # 分割数据
    batch_index = []

    scaler_for_x = MinMaxScaler(feature_range=(0, 1))  # 最小最大值标准化，将数据转换成百分比
    scaler_for_y = MinMaxScaler(feature_range=(0, 1))
    # scaled_x_data = scaler_for_x.fit_transform(data[:, :-1])  # X
    # scaled_y_data = scaler_for_y.fit_transform(data[:, -1:]).reshape(-1)  # Y
    scaled_x_data = data[:, :-1]  # X
    scaled_y_data = data[:, -1:].reshape(-1)  # Y

    label_train = scaled_y_data[:train_end]
    label_test = scaled_y_data[train_end:]
    normalized_train_data = scaled_x_data[: train_end]
    normalized_test_data = scaled_x_data[train_end:]
    # 以上获得归一化后的x和y的训练和测试集

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
def getData(num = 10,batch_size=80, time_step=15, percentage=0.7):
    batch_index= []
    train_x= []
    train_y= []
    test_x= []
    test_y = []
    action_list = []
    id = [0]
    for i in range(num):
        data = load_data(PATH['input'] + FILES[i], [DIM_STATE + DIM_ACTION, DIM_TIME])
        batch_index_t, train_x_t, train_y_t, test_x_t, test_y_t = preprocess(data, batch_size, time_step,
                                                                                 percentage)
        # 先从每一个文件得到原始数据，然后放入总数组中
        test_x.extend(test_x_t)
        test_y.extend(test_y_t)
        id.append(id[-1]+len(test_y_t))
        train_x.extend(train_x_t)
        train_y.extend(train_y_t)
        # 下面正确存储了
        if len(batch_index)>0:
            for j in range(len(batch_index_t)):
                batch_index_t[j] += batch_index[-1]
        batch_index.extend(batch_index_t)
    t_a_l = np.asarray(test_x)[:,:,DIM_STATE:]
    for t in range(t_a_l.shape[0]):
        for tt in range(t_a_l.shape[1]):
            for ttt in range(t_a_l.shape[2]):
                if t_a_l[t][tt][ttt] == 1:
                    action_list.append(ttt)
                    break
    action_list = np.asarray(action_list).reshape([-1,time_step])
    len_train_x = len(train_x)
    train_x.extend(test_x)
    train_y.extend(test_y)

    train_x = np.asarray(train_x).reshape([-1,DIM_STATE + DIM_ACTION])
    train_y = np.asarray(train_y).reshape([-1,DIM_TIME])

    scaler_for_x = MinMaxScaler(feature_range=(0, 1))  # 最小最大值标准化，将数据转换成百分比
    scaler_for_y = MinMaxScaler(feature_range=(0, 1))
    scaled_x_data = scaler_for_x.fit_transform(train_x)  # X
    scaled_y_data = scaler_for_y.fit_transform(train_y).reshape(-1)  # Y

    scaled_x_data = scaled_x_data.reshape([-1,time_step, DIM_STATE + DIM_ACTION])
    scaled_y_data = scaled_y_data.reshape([-1,time_step, DIM_TIME])

    train_x_final = scaled_x_data[:len_train_x]
    train_y_final = scaled_y_data[:len_train_x]
    test_x_final = scaled_x_data[len_train_x:]
    test_y_final = scaled_y_data[len_train_x:]

    print("训练和测试数据长度为：",train_x_final.shape[0],test_x_final.shape[0])
    return batch_index, train_x_final, train_y_final, test_x_final, test_y_final,scaler_for_y,action_list,test_y,id



# BasicRNNCell and BasicLSTMCell
def basic_rnn_model(X, cell_name=c_BasicLSTMCell, time_step=15):
    print_important('>>>> using basic_rnn_model now...')
    batch_size = tf.shape(X)[0]

    w_in = tf.Variable(tf.random_normal([INPUT_SIZE, RNN_UNIT]))
    b_in = tf.Variable(tf.constant(0.1, shape=[RNN_UNIT, ]))
    w_out = tf.Variable(tf.random_normal([RNN_UNIT, OUTPUT_SIZE]))
    b_out = tf.Variable(tf.constant(0.1, shape=[1, ]))

    cell = cell_name(RNN_UNIT)
    init_state = cell.zero_state(batch_size, dtype=tf.float32)

    input = tf.reshape(X, [-1, INPUT_SIZE])
    input_rnn = tf.matmul(input, w_in) + b_in
    input_rnn = tf.reshape(input_rnn, [-1, time_step, RNN_UNIT])
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
    output = tf.reshape(output_rnn, [-1, RNN_UNIT])
    pred = tf.matmul(output, w_out) + b_out
    """
    X->神经网络转换成大小为10的数据->RNN->输出结果
    """
    return pred, output, final_states


def sain_rnn_model(X, cell_name=None, time_step=15):
    print_important('>>>> using sain_rnn_model now...')
    batch_size = tf.shape(X)[0]

    X_state = X[:, :, :DIM_STATE]
    X_action = X[:, :, DIM_STATE:]

    w_rnn2_action = tf.Variable(tf.random_normal([DIM_ACTION, DIM_ACTION]))
    w_rnn2 = tf.Variable(tf.random_normal([DIM_STATE, DIM_ACTION]))
    b_rnn2 = tf.Variable(tf.constant(0.1, shape=[DIM_ACTION, ]))
    w_out1 = tf.Variable(tf.random_normal([DIM_ACTION, OUTPUT_SIZE]))
    w_out2 = tf.Variable(tf.random_normal([DIM_ACTION, OUTPUT_SIZE]))
    w_out3 = tf.Variable(tf.random_normal([DIM_ACTION, OUTPUT_SIZE]))
    b_out = tf.Variable(tf.constant(0.1, shape=[OUTPUT_SIZE, ]))

    cell = c_LaFeeCell(DIM_STATE, activation=math_ops.sigmoid,name='state_lafee')
    cell2 = c_LaFeeCell(DIM_ACTION, activation=math_ops.sigmoid,name='action_lafee')

    init_state = cell.zero_state(batch_size, dtype=tf.float32)
    init_state2 = cell2.zero_state(batch_size, dtype=tf.float32)

    sigmoid = math_ops.sigmoid
    add = math_ops.add
    multiply = math_ops.multiply

    rnn, satisfaction = tf.nn.dynamic_rnn(cell, X_state, initial_state=init_state, dtype=tf.float32)
    """
    output里面，包含了所有时刻的输出 H；
    state里面，包含了最后一个时刻的输出 H 和 C；
    即satisfaction只存储了最后一次的satisfaction！！！！！！！！
    问题：satisfaction中的rnn里出现了很多负值，导致log时候出现nan的结果，如何解决？
    """
    rnn = tf.reshape(rnn, [-1, DIM_STATE])
    X_action = tf.reshape(X_action, [-1, DIM_ACTION])
    # asp_sub1 = multiply(tf.matmul(math_ops.log(rnn), w_rnn2), X_action)  # sat2action
    asp_sub1 = multiply(tf.matmul(rnn, w_rnn2), X_action)  # sat2action
    asp = sigmoid(add(tf.matmul(X_action, w_rnn2_action), add(asp_sub1, b_rnn2)))
    asp = tf.reshape(asp, [-1, time_step, DIM_ACTION])
    rnn2, aspiration = tf.nn.dynamic_rnn(cell2, asp, initial_state=init_state2, dtype=tf.float32)

    rnn2 = tf.reshape(rnn2, [-1, DIM_ACTION])  # rnn2 = aspiration
    pred = add(tf.matmul(rnn2, w_out1),
               add(tf.matmul(X_action, w_out2), add(tf.matmul(multiply(rnn2, X_action), w_out3), b_out)))
    return pred, satisfaction, aspiration,rnn,rnn2,asp



def test(sess, pred, output, state, t_X, t_Y, pl_X, scaler):
    print_important('>>>> testing...')
    test_predict = []
    test_output = []
    test_state = []

    for step in range(len(t_X)):
        prob, output_, state_ = sess.run([pred, output, state], feed_dict={pl_X: [t_X[step]]})
        predict = prob.reshape((-1))
        test_predict.extend(predict)
        test_output.extend(output_)
        test_state.extend(state_)

    test_predict = scaler.inverse_transform(np.array(test_predict).reshape(-1, 1))  # 将归一化的数据还原成原来的数据
    test_y = scaler.inverse_transform(np.array(t_Y).reshape(-1, 1))
    mae = mean_absolute_error(y_pred=test_predict, y_true=test_y)
    rmse = np.sqrt(mean_squared_error(test_predict, test_y))
    print('mae:', mae, '    rmse:', rmse)
    return test_predict, test_output, test_state,mae,rmse


def train(batch_size=80, time_step=15, percentage=0.7, model=basic_rnn_model, cell=c_BasicRNNCell,
          summary_dir=TIMESTAMP,num=10):
    print_important('>>>> training...')

    # 训练数据格式：15X15X27 15X15X1
    # 测试数据格式：1X15X27 1X15X1
    X = tf.placeholder(tf.float32, shape=[None, time_step, INPUT_SIZE])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, OUTPUT_SIZE])
    batch_index, train_x, train_y, test_x, test_y, scaler_for_y,action_list,result_list,id_list = getData(num=1) # 从库中选择num个文件合并训练
    pred, output_, state_ = model(X, cell, time_step)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))  # 求预测值与标签值的均方误差
    loss_summary = tf.summary.scalar('loss', loss)

    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    with tf.Session() as sess:

        train_log_dir = 'logs/train/' + summary_dir
        test_log_dir = 'logs/test/' + summary_dir
        os.mkdir('origin_logs/'+summary_dir)
        f1 = open("origin_logs/"+summary_dir+"action.csv",'w',newline='')
        f5 = open("origin_logs/" + summary_dir + "time.csv", 'w', newline='')
        csv_write = csv.writer(f1)
        for data in action_list.tolist():
            csv_write.writerow(data)
        csv_write = csv.writer(f5)
        for data in np.asarray(result_list).reshape([-1,time_step]).tolist():
            csv_write.writerow(data)
        f1.close()
        f5.close()
        f3 = open("origin_logs/"+summary_dir+"aspiration.csv",'w',newline='')
        f2 = open("origin_logs/"+summary_dir+"satisfaction.csv",'w',newline='')
        f4 = open("origin_logs/"+summary_dir+"error.csv",'w',newline='')
        f6 = open("origin_logs/" + summary_dir + "loss.csv", 'w', newline='')
        f7 = open("origin_logs/" + summary_dir + "user.csv", 'w', newline='')
        csv_write2 = csv.writer(f2)
        csv_write3 = csv.writer(f3)
        csv_write4 = csv.writer(f4)
        csv_write5 = csv.writer(f6)
        csv_write6 = csv.writer(f7)
        csv_write6.writerow(id_list)# 写入id
        f7.close()
        merged = tf.summary.merge([loss_summary])
        train_writer = tf.summary.FileWriter(train_log_dir, sess.graph)
        test_writer = tf.summary.FileWriter(test_log_dir)

        sess.run(tf.global_variables_initializer())
        # 并没有对batch_index进行打乱
        for i in range(ITER_TIME):
            loss_ = None
            summary_train = None
            for step in range(len(batch_index) - 1):
                if batch_index[step] != batch_index[step+1]: # 注意这行，如果两个值相同意味着来了新用户
                    summary_train, _, loss_,pred_, = sess.run([merged, train_op, loss,pred,],
                                                   feed_dict={X: train_x[batch_index[step]: batch_index[step + 1]],
                                                              Y: train_y[batch_index[step]: batch_index[step + 1]]})
                    csv_write5.writerow([loss_])
            if i % 100 == 0:
                print('iter:', i, 'loss:', loss_)
                test_predict, test_output, test_state,mae,rmse = test(sess, pred, output_, state_, test_x, test_y, X,
                                                             scaler_for_y)
                # print("test_predict:", test_predict)
                csv_write4.writerow([mae,rmse]) # 记录error
                for data in np.asarray(test_output).tolist():
                    csv_write2.writerow(data)
                for data in np.asarray(test_state).tolist():
                    csv_write3.writerow(data)


                # print("test_state(aspiration):", test_state)

            train_writer.add_summary(summary_train, i)
        f2.close()
        f3.close()


    train_writer.close()
    test_writer.close()
    return test_predict, test_output, test_state


if __name__ == '__main__':
    # getData(10)
    # data = load_data(PATH['input'] + FILES[0], [DIM_STATE + DIM_ACTION, DIM_TIME])
    # with tf.variable_scope('sain_rnn_model'):
    #     summary_dir = TIMESTAMP + '_sain_rnn_model/'
    #     test_predict, test_output, test_state = train(model=sain_rnn_model, summary_dir=summary_dir)
    # 使用RNN和LSTM直接测试
    with tf.variable_scope('basic_rnn_cell'):
        summary_dir = TIMESTAMP + '_basic_rnn_model/'
        train(model=basic_rnn_model, cell=c_BasicRNNCell, summary_dir=summary_dir)
    with tf.variable_scope('basic_lstm_cell'):
        summary_dir = TIMESTAMP + '_basic_lstm_model/'
        train(model=basic_rnn_model, cell=c_BasicLSTMCell, summary_dir=summary_dir)
"""
1. model（RF,SVM,DNN,RNN,LSTM与LaFee)对比  loss训练集曲线以及mae,rmse的测试集结果
2. 不同登出时间对应的满意度的分布（统计所有用户登出时间histogram，实际登出时间）
3. colormap 同一个用户不相似的行为序列以及两个用户相似的行为序列 比较他们aspiration的colormap 15X19
"""