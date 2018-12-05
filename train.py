import tensorflow as tf
import datainput
import fx
import network

TRAIN_STEP = 1000000
#DATA_SIZE = 2400
BATCH_SIZE = 1000

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 18])
ys = tf.placeholder(tf.float32, [None, 16])

prediction = network.Network(xs).model

# the error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys - prediction)), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

data = datainput.DataInput(fx.LPCNet())

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, "./model.ckpt")
    for i in range(1, TRAIN_STEP + 1):
        x, y = data.next_batch(BATCH_SIZE)
        sess.run(train_step, feed_dict={xs: x, ys: y})
        if i % 1000 == 0:
            prediction_value, loss_value = sess.run([prediction, loss], feed_dict={xs: x, ys: y})
            print('step %s, loss=%s' % (i, loss_value))
            saver.save(sess, "D:/lpc_checkpoint/model.ckpt")




