import tensorflow as tf


def training_setup(y_, y_conv,
                   step_size=1e-4, device='/cpu:0'):
    """
    Creates the TensorFlow variables and operation required to train a
    defined neural network.

    :param y_: The 'true' result corresponding to a given input x.
    :param tf.Tensor y_conv: The result of the computation of the neural
        network given an input x.
    :param dbl step_size: The step side of the AdamOptimizer.
    :param str device: The device to use for storing variables and computation.
        Either '/cpu:<n>' or '/cpu:<n>'. Defaults to '/cpu:<n>.
    :return: A set of TensorFlow tensors and an operation that are involved in
        controlling the updating of the neural network.
    :rtype: (tf.Tensor, tf.Operation, tf.Tensor, tf.Tensor).
    """
    with tf.device(device_name_or_function=device):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
        train_step = tf.train.AdamOptimizer(step_size).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return cross_entropy, train_step, correct_prediction, accuracy