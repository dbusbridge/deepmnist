import nettools as nt
import tensorflow as tf


def multilayer_convnet(device='/cpu:0'):
    """Create a multilayer convolutional neural network.

    :param str device: The device to use for storing variables and computation.
        Either '/cpu:<n>' or '/cpu:<n>'. Defaults to '/cpu:<n>.
    :return: A set of TensorFlow tensors that serve as inputs, controllers and
        outputs of the network.
    :rtype: (tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor).
    """
    with tf.device(device_name_or_function=device):
        # Placeholders for feed and output
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y_ = tf.placeholder(tf.float32, shape=[None, 10])

        # Reshape image for convolving
        x_image = tf.reshape(x, [-1, 28, 28, 1])

        # First convolutional layer: convolution weights + bias
        W_conv1 = nt.weight_variable([5, 5, 1, 32])
        b_conv1 = nt.bias_variable([32])

        # First convolutional layer: perform convolution + bias with relu
        h_conv1 = tf.nn.relu(nt.conv2d(x_image, W_conv1) + b_conv1)

        # First convolutional layer: pooling
        h_pool1 = nt.max_pool_2x2(h_conv1)

        # Second convolutional layer: convolution weights + bias
        W_conv2 = nt.weight_variable([5, 5, 32, 64])
        b_conv2 = nt.bias_variable([64])

        # Second convolutional layer: perform convolution + bias with relu
        h_conv2 = tf.nn.relu(nt.conv2d(h_pool1, W_conv2) + b_conv2)

        # Second convolutional layer: pooling
        h_pool2 = nt.max_pool_2x2(h_conv2)

        # Flatten pooled ready for linear transformation
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

        # First fully connected layer: linear transformation weights + bias
        W_fc1 = nt.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = nt.bias_variable([1024])

        # First fully connected layer: perform linear transformation
        # + bias with relu
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Dropout layer to prevent overfitting
        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # Readout layer: linear transformation weights + bias
        W_fc2 = nt.weight_variable([1024, 10])
        b_fc2 = nt.bias_variable([10])

        # Readout layer: perform linear transformation + bias with relu
        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return x, y_, y_conv, keep_prob

