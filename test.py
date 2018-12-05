import tensorflow as tf
import numpy as np
import network
import datainput
import fx

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 18])
ys = tf.placeholder(tf.float32, [None, 16])

prediction = network.Network(xs).model

# the error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys - prediction)), reduction_indices=[1]))

saver = tf.train.Saver()
data = datainput.DataInput(fx.LPCNet())
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "D:/lpc_checkpoint/model.ckpt")
    x, y = data.next_batch(1)
    prediction_value = sess.run(prediction, feed_dict={xs: x})
    print(prediction_value)
    print(np.array(y))
    print(np.mean(prediction_value - y))



