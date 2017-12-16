import tensorflow as tf
import numpy as np


def weight_def(shape, stddev = 0.1, name = 'undefined'):
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name = name)


def bias_def(shape, name):
    return tf.Variable(tf.constant(0.1, shape=shape))





def conv2d_dropout_relu(x, W, B, keep_prob, name):
    conv2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)
    dropout = tf.nn.dropout(conv2d, keep_prob)
    return tf.nn.relu(tf.nn.bias_add(dropout, B))


def maxpool2d(x, name = 'undefined'):
    """ ksize defines the size of the pool window
     strides defines the way the window moves (movement of the window)
     """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = name)



def deconv2d_relu(x, W, B, name):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]])
    deconv = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, 2, 2, 1], padding='SAME')
    return tf.nn.relu(tf.nn.bias_add(deconv, B))







# Copy pasted functions
def pixel_wise_softmax_2(output_map):
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, 3, keep_dims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, tf.shape(output_map)[3]]))
    return tf.div(exponential_map,tensor_sum_exp)



def cross_entropy(y_,output_map):
    return -tf.reduce_mean(y_*tf.log(tf.clip_by_value(output_map,1e-10,1.0)), name="cross_entropy")





def accuracy(predictions, labels):
    """
    Return the error rate based on dense predictions and 1-hot labels.
    """

    return (100.0 * np.sum(np.argmax(predictions, 3) == np.argmax(labels, 3)) /
                (predictions.shape[0]*predictions.shape[1]*predictions.shape[2]))
