import tensorflow as tf
import numpy as np


def conv2d_relu(x, W, B, name = 'undefined'):
    conv2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME', name=name)
    return tf.nn.relu(tf.nn.bias_add(conv2d, B))


def maxpool2d(x, name = 'undefined'):
    """ ksize defines the size of the pool window
     strides defines the way the window moves (movement of the window)
     """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = name)

def deconv2d_relu(x, W, B, upscale_factor, name = 'undefined'):
    stride = upscale_factor #upscale_factor
    strides = [1, stride, stride, 1]
    # Shape of the x tensor
    in_shape = tf.shape(x)
    i = 0
    if (in_shape[1] < 20):
        i=1
    h = ((in_shape[1] - 1) * stride) + 2 - i #1*(in_shape[1] == 13)#(in_shape[1]%2)
    w = ((in_shape[2] - 1) * stride) + 2 - i #1*(in_shape[2] == 13)#(in_shape[2]%2)
    new_shape = [in_shape[0], h, w, W.shape[3]]
    output_shape = tf.stack(new_shape)
    deconv = tf.nn.conv2d_transpose(x, W, output_shape,
                                    strides=strides, padding='SAME', name = name)
    return tf.nn.relu(tf.nn.bias_add(deconv, B))

def bn_conv_relu(x, W, B, beta, gamma, name): # By default 64.
    bn = batchnorm2d(x, beta, gamma, 'bn' + name)
    relu = conv2d_relu(bn, W, B, name='deconv' + name)
    # conv = conv2d(bn, tf.Variable(tf.truncated_normal([3, 3, in_channels, out_channels])))
    # relu = tf.nn.relu(tf.nn.bias_add(conv, tf.Variable(tf.truncated_normal([out_channels]))))
    return relu

def bn_deconv_relu(x, W, B, beta, gamma, upscale_factor, name): # By default 64.
    bn = batchnorm2d(x, beta, gamma, 'bn' + name)
    relu = deconv2d_relu(bn, W, B, upscale_factor=upscale_factor, name='deconv' + name)
    # conv = conv2d(bn, tf.Variable(tf.truncated_normal([3, 3, in_channels, out_channels])))
    # relu = tf.nn.relu(tf.nn.bias_add(conv, tf.Variable(tf.truncated_normal([out_channels]))))
    return relu

def batchnorm2d(x, beta, gamma, name='undefined'):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments' + name)
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update(batch_mean, batch_var):
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # mean, var = tf.cond(mean_var_with_update(batch_mean, batch_var),
        #                     lambda: ema.average(batch_mean), lambda: ema.average(batch_var)) #IDK

        #tf.nn.moments()
        normed = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-3, name=name)
    return normed








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
