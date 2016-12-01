import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as tfid
import networks
import trainer

# Device to use, either '/cpu:<x>' or '/gpu:<x>'
DEVICE = '/cpu:0'
# DEVICE = '/gpu:0'

# Start the session
sess = tf.InteractiveSession()

# Read in the data
mnist = tfid.read_data_sets('MNIST_data', one_hot=True)

# Construct the neural network
x, y_, y_conv, keep_prob = networks.multilayer_convnet(device=DEVICE)

# Define the training step
(cross_entropy, train_step,
 correct_prediction, accuracy) = trainer.training_setup(
    y_=y_, y_conv=y_conv, device=DEVICE)

# The session
sess.run(tf.global_variables_initializer())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(
            feed_dict={
                x: batch[0],
                y_: batch[1],
                keep_prob: 1.0})
        print("step {i}, training accuracy {acc}".format(i=i,
                                                         acc=train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# Print out accuracy on the training set
print("test accuracy {acc}".format(
    acc=accuracy.eval(feed_dict={
        x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
