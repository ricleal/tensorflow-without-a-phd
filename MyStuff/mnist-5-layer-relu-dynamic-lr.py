import math
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

'''
Simplest model possible:
784 (image = 28 x 28 ) input nodes, 10 output nodes.
No hidden layers
'''

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# None because of the batch size. We don't know yet its size!
X = tf.placeholder(tf.float32, [None, 784])

# placeholder for correct labels (correct answers will go here)
Y_ = tf.placeholder(tf.float32, [None, 10])

# five layers and their number of neurons (the last layer has 10 softmax neurons)
K = 200
L = 100
M = 60
N = 30

# Weights initialised with small random values between -0.2 and +0.2
# When using RELUs, make sure biases are initialised with small *positive* values for example 0.1 = tf.ones([K])/10
W1 = tf.Variable(tf.truncated_normal([784, K], stddev=0.1))  # 784 = 28 * 28
B1 = tf.Variable(tf.zeros([K]))

W2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
B2 = tf.Variable(tf.zeros([L]))

W3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B3 = tf.Variable(tf.zeros([M]))

W4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B4 = tf.Variable(tf.zeros([N]))

W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))


# The model
X = tf.reshape(X, [-1, 784])
Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y3 = tf.nn.relu(tf.matmul(Y2, W3) + B3)
Y4 = tf.nn.relu(tf.matmul(Y3, W4) + B4)
Y = tf.nn.softmax(tf.matmul(Y4, W5) + B5)

# loss function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# step for variable learning rate
step = tf.placeholder(tf.int32)
# the learning rate is: # 0.0001 + 0.003 * (1/e)^(step/2000)), i.e. exponential decay from 0.003->0.0001
lr = 0.0001 +  tf.train.exponential_decay(0.003, step, 2000, 1/math.e)
optimizer = tf.train.GradientDescentOptimizer(lr)
train_step = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
n_batches = 3000
for i in range(n_batches):
    # load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(batch_size)
    train_data = {X: batch_X, Y_: batch_Y, step: i}

    # train
    sess.run(train_step, feed_dict=train_data)

    # success ?
    a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)

    # success on test data ?
    if i % 100 == 0 or i == n_batches - 1: 
        test_data = {X: mnist.test.images, Y_: mnist.test.labels}
        a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
        print("** Accuracy on the Training Dataset =", a)
