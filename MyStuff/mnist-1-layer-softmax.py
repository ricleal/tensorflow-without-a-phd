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
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# model
Y = tf.nn.softmax(tf.matmul(X, W) + b)

# placeholder for correct labels
Y_ = tf.placeholder(tf.float32, [None, 10])

# loss function
cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))

# % of correct answers found in batch
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
n_batches = 3000
for i in range(n_batches):
    # load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(batch_size)
    train_data = {X: batch_X, Y_: batch_Y}

    # train
    sess.run(train_step, feed_dict=train_data)

    # success ?
    a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)

    # success on test data ?
    if i % 100 == 0 or i == n_batches - 1: 
        test_data = {X: mnist.test.images, Y_: mnist.test.labels}
        a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
        print("** Accuracy on the Training Dataset =", a)
