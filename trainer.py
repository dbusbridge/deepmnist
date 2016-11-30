import tensorflow as tf


def training_setup(y_, y_conv,
                   step_size=1e-4, device='/cpu:0'):
    """

    :param y_:
    :param y_conv:
    :param step_size:
    :param device:
    :return:
    :rtype:
    """
    with tf.device(device_name_or_function=device):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
        train_step = tf.train.AdamOptimizer(step_size).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return cross_entropy, train_step, correct_prediction, accuracy