import tensorflow as tf


def network(x):
    fc1 = fc_layer(x, 288, "fc1")
    ac1 = tf.nn.leaky_relu(fc1)
    fc2 = fc_layer(ac1, 288, "fc2")
    ac2 = tf.nn.sigmoid(fc2)
    fc3 = fc_layer(ac2, 288, "fc3")
    ac3 = tf.nn.sigmoid(fc3)
    fc4 = fc_layer(ac3, 288, "fc4")
    ac4 = tf.nn.sigmoid(fc4)
    out = fc_layer(ac4, 16, "out")
    return out


def fc_layer(bottom, n_weight, name):
    assert len(bottom.get_shape()) == 2
    n_prev_weight = bottom.get_shape()[1]
    W = tf.get_variable(name + 'W',
                        dtype=tf.float32,
                        shape=[n_prev_weight, n_weight],
                        initializer=tf.truncated_normal_initializer(mean=0., stddev=0.2))
    b = tf.get_variable(name + 'b',
                        dtype=tf.float32,
                        initializer=tf.constant(0, shape=[n_weight], dtype=tf.float32))
    fc = tf.nn.bias_add(tf.matmul(bottom, W), b)
    return fc


class Network:
    def __init__(self, x):
        self.model = network(x)

