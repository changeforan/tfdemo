import tensorflow as tf
import matplotlib.pyplot as plt
import datainput
import fx


def network(x):
    fc1 = fc_layer(x, 10, "fc1")
    ac1 = tf.nn.relu(fc1)
    fc2 = fc_layer(ac1, 20, "fc2")
    ac2 = tf.nn.sigmoid(fc2)
    fc3 = fc_layer(ac2, 20, "fc3")
    ac3 = tf.nn.sigmoid(fc3)
    fc4 = fc_layer(ac3, 1, "fc4")
    return fc4


def fc_layer(bottom, n_weight, name):
    assert len(bottom.get_shape()) == 2
    n_prev_weight = bottom.get_shape()[1]
    W = tf.get_variable(name+'W',
                        dtype=tf.float32,
                        shape=[n_prev_weight, n_weight],
                        initializer=tf.truncated_normal_initializer(stddev=0.5))
    b = tf.get_variable(name+'b',
                        dtype=tf.float32,
                        initializer=tf.constant(0.1,
                                                shape=[n_weight],
                                                dtype=tf.float32))
    fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
    return fc


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

prediction = network(xs)

# the error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

data = datainput.DataInput(fx.PiFunction(100))
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(16000):
        x, y = data.next_batch(20)
        sess.run(train_step, feed_dict={xs: x, ys: y})
    prediction_value = sess.run(prediction, feed_dict={xs: data.x_})
    lines = ax.plot(data.x_, prediction_value, 'ro', lw=3)


ax.scatter(data.x_, data.y_)
plt.show()
