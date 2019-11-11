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
from c_lstm import print_important

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

# Read data from files and divide the dimensions
def load_data(file_path, dim_list = [DIM_STATE + DIM_ACTION, DIM_TIME], split=False):
    f = open(file_path)
    print("file_path:",file_path)
    df = pd.read_csv(f,header=None) # 首行也算！！！！
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
def getData(num = 10,batch_size=80, time_step=15, percentage=0.7):
    batch_index= []
    train_x= []
    train_y= []
    test_x= []
    test_y = []
    action_list = []
    is_logout_train = []
    is_logout_test = []
    id = [0]
    for i in range(num):
        # 100140101
        data = load_data(PATH['input'] + FILES[i])
        # data = load_data(PATH['input'] + "100140101.csv")
        batch_index_t, train_x_t, train_y_t, test_x_t, test_y_t = preprocess(data, batch_size, time_step,
                                                                                 percentage)
        # 先从每一个文件得到原始数据，然后放入总数组中
        test_x.extend(test_x_t)
        test_y.extend(test_y_t)
        id.append(id[-1]+len(test_y_t)) # record the index of each test batch
        train_x.extend(train_x_t)
        train_y.extend(train_y_t)
        # 下面正确存储了
        if len(batch_index)>0:
            for j in range(len(batch_index_t)):
                batch_index_t[j] += batch_index[-1] # adjust the batch index with offset
        batch_index.extend(batch_index_t)
    t_a_l = np.asarray(test_x)[:, :, DIM_STATE:]
    t_a_t = np.asarray(train_x)[:, :, DIM_STATE:]
    for t in range(t_a_l.shape[0]):
        for tt in range(t_a_l.shape[1]):
            for ttt in range(t_a_l.shape[2]):
                if t_a_l[t][tt][ttt] == 1: # find the action executed this time
                    action_list.append(ttt) # save the actions with their positions in order
                    if ttt == 1: # judge whether the action is logout (index of logout is 1)
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
    return batch_index, train_x_final, train_y_final, test_x_final, test_y_final, scaler_for_y, action_list, test_y, id, is_logout_test, is_logout_train

"""
这里应该只用最后一次的satisfaction和aspiration才对
问席萌？？
"""
def lafee_model(X,ACTION):

    print_important('>>>> using lafee_model now...')
    batch_size = tf.shape(X)[0]

    X_state = X[:, :, :DIM_STATE]
    X_action = X[:, :, DIM_STATE:]

    with tf.variable_scope('rnn_satisfaction'):
        satis_cell = c_LaFeeCell(DIM_STATE, activation=math_ops.sigmoid, name='satis_cell')
        # satis_cell = c_BasicLSTMCell(DIM_STATE, activation=math_ops.sigmoid, name='satis_cell')
        init_state = satis_cell.zero_state(batch_size, dtype=tf.float32)
        rnn_satis_, satisfaction = tf.nn.dynamic_rnn(satis_cell, X_state, initial_state=init_state, dtype=tf.float32)
        rnn_satis = tf.reshape(rnn_satis_, [-1, DIM_STATE])  # !!!!注意这里的变量使用的是后面的
    with tf.variable_scope('rnn_aspiration'):
        aspir_cell = c_LaFeeCell(DIM_ACTION, activation=math_ops.sigmoid, name='aspir_cell')
        # aspir_cell = c_BasicLSTMCell(DIM_ACTION, activation=math_ops.sigmoid, name='aspir_cell')
        init_state2 = aspir_cell.zero_state(batch_size, dtype=tf.float32)
        rnn_aspir_, aspiration = tf.nn.dynamic_rnn(aspir_cell, X_action, initial_state=init_state2, dtype=tf.float32)
        rnn_aspir = tf.reshape(rnn_aspir_, [-1, DIM_ACTION])  # !!!!注意这里的变量使用的是后面的

    X_action_new = tf.reshape(X_action, [-1, DIM_ACTION])
    asp_sat = tf.concat([rnn_satis, rnn_aspir], 1)  # 将得到的aspiration和satisfaction拼接在一起


    # 假设这里层数和网络参数变少
    Multi = True # 多参数开关
    if Multi:
        print("multi-network")
        aspiration_l1 = tf.layers.dense(asp_sat, 200, tf.nn.relu, name='aspiration_l1')  # hidden layer 输入层增加方式
        aspiration_l2 = tf.layers.dense(aspiration_l1, 100, tf.nn.relu, name='aspiration_l2')  # hidden layer 输入层增加方式
        aspiration_output = tf.layers.dense(aspiration_l2, DIM_ACTION, name='aspiration_output')  # output layer 输出层增加方式
        asp_act = tf.concat([aspiration_output, X_action_new], 1)
        tin_l1 = tf.layers.dense(asp_act, 200, tf.nn.relu, name='tin_l1')  # hidden layer 输入层增加方式
        tin_l2 = tf.layers.dense(tin_l1, 100, tf.nn.relu, name='tin_l2')  # hidden layer 输入层增加方式
        tin = tf.layers.dense(tin_l2, 1, name='tin')  # output layer 输出层增加方式
        satisfaction_l1 = tf.layers.dense(asp_sat, 200, tf.nn.relu, name='satisfaction_l1')  # hidden layer 输入层增加方式
        satisfaction_l2 = tf.layers.dense(satisfaction_l1, 100, tf.nn.relu, name='satisfaction_l2')  # hidden layer 输入层增加方式
        satisfaction_output = tf.layers.dense(satisfaction_l2, DIM_STATE,name='satisfaction_output')  # output layer 输出层增加方式
        tout_l1 = tf.layers.dense(satisfaction_output, 200, tf.nn.relu, name='tout_l1')  # hidden layer 输入层增加方式
        tout_l2 = tf.layers.dense(tout_l1, 100, tf.nn.relu, name='tout_l2')  # hidden layer 输入层增加方式
        tout = tf.layers.dense(tout_l2, 1, name='tout')  # output layer 输出层增加方式
    else:
        print("less-network")
        aspiration_output = tf.layers.dense(asp_sat, DIM_ACTION, name='aspiration_output')  # output layer 输出层增加方式
        asp_act = tf.concat([aspiration_output, X_action_new], 1)
        tin = tf.layers.dense(asp_act, 1, name='tin')  # output layer 输出层增加方式
        satisfaction_output = tf.layers.dense(asp_sat, DIM_STATE,name='satisfaction_output')  # output layer 输出层增加方式
        tout = tf.layers.dense(satisfaction_output, 1, name='tout')  # output layer 输出层增加方式

    ACTION_ = tf.reshape(ACTION, [-1, 1])
    pred = tf.where(ACTION_,tout,tin)
    satisfaction_output = tf.reshape(tf.reshape(satisfaction_output,[-1,15,DIM_STATE])[:,-1,:],[-1,DIM_STATE])
    aspiration_output = tf.reshape(tf.reshape(aspiration_output,[-1,15,DIM_ACTION])[:,-1,:],[-1,DIM_ACTION])
    return pred, satisfaction_output, aspiration_output # 返回的是过了神经网络之后的satisfaction和aspiration


def test_model(sess, pred, satisfaction_output, aspiration_output, t_X, t_Y, pl_X, scaler,actionlist,ACTION):
    print_important('>>>> testing...')
    test_predict = []
    test_output = []
    test_state = []

    for step in range(len(t_X)):
        prob, output_, state_ = sess.run([pred, satisfaction_output, aspiration_output], feed_dict={pl_X: [t_X[step]],ACTION:[actionlist[step]]})
        predict = prob.reshape((-1))
        test_predict.extend(predict)
        test_output.extend(output_)
        test_state.extend(state_)

    test_predict = scaler.inverse_transform(np.array(test_predict).reshape(-1, 1))  # 将归一化的数据还原成原来的数据
    test_y = scaler.inverse_transform(np.array(t_Y).reshape(-1, 1))
    mae = mean_absolute_error(y_pred=test_predict, y_true=test_y)
    rmse = np.sqrt(mean_squared_error(test_predict, test_y))
    test_y = np.reshape(np.reshape(test_y,[-1,15,1])[:,-1,:],[-1,1]) # 只保留15步最后一步的预测值
    test_predict = np.reshape(np.reshape(test_predict,[-1,15,1])[:,-1,:],[-1,1]) # 只保留15步最后一步的预测值
    print('mae:', mae, '    rmse:', rmse)
    return test_predict, test_output, test_state, mae, rmse,test_y


def train(batch_size=80, time_step=15, percentage=0.7, model=lafee_model, cell=c_BasicRNNCell,
          summary_dir=TIMESTAMP, num=1):
    print_important('>>>> training...')

    # 训练数据格式：15X15X27 15X15X1
    # 测试数据格式：1X15X27 1X15X1
    X = tf.placeholder(tf.float32, shape=[None, time_step, INPUT_SIZE])
    Y = tf.placeholder(tf.float32, shape=[None, time_step, OUTPUT_SIZE])
    ACTION= tf.placeholder(tf.bool,shape=[None,time_step,1]) # 判断行为是哪个
    batch_index, train_x, train_y, test_x, test_y, scaler_for_y, action_list, result_list, id_list, is_logout_test, is_logout_train = getData(
        num=num)  # 从库中选择num个文件合并训练
    is_logout_test = np.asarray(is_logout_test).reshape([-1,time_step,1])
    is_logout_train = np.asarray(is_logout_train).reshape([-1,time_step,1])
    pred, satisfaction_output, aspiration_output = model(X,ACTION)
    loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))  # 求预测值与标签值的均方误差
    loss_summary = tf.summary.scalar('loss', loss)

    train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

    with tf.Session() as sess:

        train_log_dir = 'logs/train/' + summary_dir
        test_log_dir = 'logs/test/' + summary_dir
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
        f7 = open("origin_logs/" + summary_dir + "user.csv", 'w', newline='')
        f8 = open("origin_logs/" + summary_dir + "pred.csv", 'w', newline='')
        csv_write2 = csv.writer(f2)
        csv_write3 = csv.writer(f3)
        csv_write4 = csv.writer(f4)
        csv_write6 = csv.writer(f6)
        csv_write7 = csv.writer(f7)
        csv_write8 = csv.writer(f8)
        csv_write7.writerow(id_list)  # 写入id
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
                if batch_index[step] != batch_index[step + 1]:  # 注意这行，如果两个值相同意味着来了新用户

                    summary_train, _, loss_, pred_ = sess.run(
                        [merged, train_op, loss, pred],
                        feed_dict={X: train_x[batch_index[step]: batch_index[step + 1]],
                                   Y: train_y[batch_index[step]: batch_index[step + 1]],
                                   ACTION:is_logout_train[batch_index[step]: batch_index[step + 1]]})
                    csv_write6.writerow([loss_])
            if i % 100 == 0:
                print('iter:', i, 'loss:', loss_)
                test_predict, test_output, test_state, mae, rmse, test_yy = test_model(sess, pred, satisfaction_output, aspiration_output, test_x, test_y, X,scaler_for_y,is_logout_test, ACTION)
                # print("test_predict:", test_predict)
                csv_write4.writerow([mae, rmse])  # 记录error
                for data in np.asarray(test_output).tolist():
                    csv_write2.writerow(data) # satisfaction
                for data in np.asarray(test_state).tolist():
                    csv_write3.writerow(data) # aspiration
                pred_y = np.concatenate([test_predict,test_yy],axis=1)
                for data in np.asarray(pred_y).tolist():
                    csv_write8.writerow(data) # aspiration
                # print("test_state(aspiration):", test_state)

            train_writer.add_summary(summary_train, i)
        f2.close()
        f3.close()
        f4.close()
        f7.close()
        f8.close()
        f6.close()

    train_writer.close()
    test_writer.close()
    return test_predict, test_output, test_state


if __name__ == '__main__':
    # getData(10)
    # data = load_data(PATH['input'] + FILES[0])
    summary_dir = TIMESTAMP + '_lafee_model/'
    test_predict, test_output, test_state = train(model=lafee_model, summary_dir=summary_dir,num=1) # num为用户读入用户数目
"""
效果好的可能原因：参数过多，后面两个神经网络参数过多(确实是这样）
就算效果很好，也要注意：代码逻辑是否正确  比较标准是否统一  以及后续实验

实际时间序列和预测时间序列的折线图 时间 = log（时间） 所有的

satisfaction & aspiration should be values calculated in lafee cells, not the satisfaction_out & aspiration_out used now.
There is something wrong with test: test data from different users were tested together ?
The id are indexes of where users' test data start
"""