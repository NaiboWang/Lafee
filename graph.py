import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)
"""
tf.where使用示例，用两个神经网络拟合两个模型
但如果两个模型X差距过大，这样做就会使得模型在局部过拟合
换adam优化器效果会好很多
"""

# fake data
x = np.linspace(-1, 1, 100)[:, np.newaxis]          # shape (100, 1)
noise = np.random.normal(0, 0.1, size=x.shape)
y = np.power(x, 2) + noise                          # shape (100, 1) + some noise
action = np.ones(100)[:, np.newaxis]
action = action.astype('bool')
for i in range(30):
    y[i*3] = x[i*3] * 2
    action[i*3] = False

tf_x = tf.placeholder(tf.float32, x.shape)     # input x
tf_y = tf.placeholder(tf.float32, y.shape)     # input y
tf_action = tf.placeholder(tf.bool,[None,1])

# neural network layers
l1 = tf.layers.dense(tf_x, 100, tf.nn.relu)          # hidden layer 输入层增加方式
output = tf.layers.dense(l1, 1)                     # output layer 输出层增加方式
l2 = tf.layers.dense(tf_x, 100, tf.nn.relu)          # hidden layer 输入层增加方式
output2 = tf.layers.dense(l2, 1)                     # output layer 输出层增加方式


pred_ = tf.where(tf_action,output,output2)

loss = tf.losses.mean_squared_error(tf_y, pred_)   # compute cost 均方误差
optimizer = tf.train.AdamOptimizer(learning_rate=0.1) # 梯度下降优化
train_op = optimizer.minimize(loss) #最小化loss

sess = tf.Session()                                 # control training and others
sess.run(tf.global_variables_initializer())         # initialize var in graph
plt.ion()   # something about plotting

for step in range(1000):
    # train and net output
    _, l, pred = sess.run([train_op, loss, pred_], {tf_x: x, tf_y: y,tf_action:action})
    if step % 5 == 0:
        # plot and show learning process
        plt.cla() # close all plot
        plt.scatter(x, y)
        plt.scatter(x, pred)
        plt.text(0.5, 0, 'Loss=%.4f' % l, fontdict={'size': 20, 'color': 'red'})
        print(l)
        plt.pause(0.1)

plt.ioff()
plt.show()
