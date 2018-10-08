import tensorflow as tf
import matplotlib.pyplot as plt
import datainput
import fx


TRAIN_STEP = 100000
DATA_SIZE = 100
BATCH_SIZE = 10


def network(x):
    fc1 = fc_layer(x, 10, "fc1")
    ac1 = tf.nn.relu(fc1)
    fc2 = fc_layer(ac1, 30, "fc2")
    ac2 = tf.nn.sigmoid(fc2)
    fc3 = fc_layer(ac2, 30, "fc3")
    ac3 = tf.nn.sigmoid(fc3)
    out = fc_layer(ac3, 1, "out")
    return out


def fc_layer(bottom, n_weight, name):
    assert len(bottom.get_shape()) == 2
    n_prev_weight = bottom.get_shape()[1]
    W = tf.get_variable(name+'W',
                        dtype=tf.float32,
                        shape=[n_prev_weight, n_weight],
                        initializer=tf.truncated_normal_initializer(mean=0., stddev=0.3))
    b = tf.get_variable(name+'b',
                        dtype=tf.float32,
                        initializer=tf.constant(0, shape=[n_weight], dtype=tf.float32))
    fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
    return fc


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

prediction = network(xs)

# the error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys - prediction)), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

data = datainput.DataInput(fx.PiFunction(DATA_SIZE))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1, TRAIN_STEP + 1):
        x, y = data.next_batch(BATCH_SIZE)
        sess.run(train_step, feed_dict={xs: x, ys: y})
        if i % 1000 == 0:
            print(i)
            plt.clf()
            plt.plot(data.x_, data.y_, lw=4)
            prediction_value = sess.run(prediction, feed_dict={xs: data.x_})
            plt.plot(data.x_, prediction_value, lw=2)
            plt.show()




