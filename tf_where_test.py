import tensorflow as tf
import numpy as np
x = [[1],[4]]
y = [[7],[12]]
condition3 = [[True],[False]]
with tf.Session() as sess:
    print(np.asarray(x).shape)
    print(np.asarray(condition3).shape)
    print(sess.run(tf.where(condition3,x,y)))

