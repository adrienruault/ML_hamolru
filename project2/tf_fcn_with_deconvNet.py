
import numpy as np
import tensorflow as tf
import preprocessing as pre
from random import shuffle

# To be changed to 2
NUM_CLASSES = 2

# To be changed to 400, 400
IMG_WIDTH = 400
IMG_HEIGHT = 400

NUM_EPOCHS = 2

# To be changed to 3
NUM_CHANNELS = 3

# To be changed to 1
BATCH_SIZE = 1

TRAIN_SIZE = 5
TEST_SIZE = 10

MODEL = 0

data = pre.load_images("data/training/images/", TRAIN_SIZE)
labels = pre.load_groundtruths("data/training/groundtruth/", TRAIN_SIZE)
#test_data  = pre.load_images("./data/test_set_images/")

# These values need to be fed in the sess.run() function using feed_dict
x = tf.placeholder('float', [BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, 3])
y = tf.placeholder('float', [BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, 1])


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool2d(x):
    """ ksize defines the size of the pool window
     strides defines the way the window moves (movement of the window)
     """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


def batchnorm2d(x, n_out, phase_train):
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
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update(batch_mean, batch_var):
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        # mean, var = tf.cond(mean_var_with_update(batch_mean, batch_var),
        #                     lambda: ema.average(batch_mean), lambda: ema.average(batch_var)) #IDK

        #tf.nn.moments()
        normed = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, gamma, 1e-3)
    return normed


def bn_conv_relu(x, in_channels=64, out_channels=64): # By default 64.
    ph =tf.placeholder(tf.bool, name='phase_train')
    bn = batchnorm2d(x, in_channels, x== x)
    conv = conv2d(bn, tf.Variable(tf.truncated_normal([3, 3, in_channels, out_channels])))
    relu = tf.nn.relu(tf.nn.bias_add(conv, tf.Variable(tf.truncated_normal([out_channels]))))
    return relu


def bn_upconv_relu(x, in_channels=64, upscale_factor=2): # By default 64.
    bn = batchnorm2d(x, in_channels, False)
    conv = upsample_layer(bn, in_channels, upscale_factor)
    relu = tf.nn.relu(tf.nn.bias_add(conv, tf.Variable(tf.truncated_normal([in_channels]))))
    return relu


def upsample_layer(bottom,
                   n_channels, upscale_factor):
    kernel_size = 3 #2 * upscale_factor - upscale_factor % 2
    stride = upscale_factor #upscale_factor
    strides = [1, stride, stride, 1]
    with tf.variable_scope("upconv"):
        # Shape of the bottom tensor
        in_shape = tf.shape(bottom)
        h = ((in_shape[1] - 1) * stride) + 2
        w = ((in_shape[2] - 1) * stride) + 2
        new_shape = [in_shape[0], h, w, n_channels]
        output_shape = tf.stack(new_shape)
        filter_shape = [kernel_size, kernel_size, n_channels, n_channels]
        weights = get_bilinear_filter(filter_shape, upscale_factor)
        deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                        strides=strides, padding='SAME')
    return deconv


def get_bilinear_filter(filter_shape, upscale_factor):
    ##filter_shape is [width, height, num_in_channels, num_out_channels]
    kernel_size = filter_shape[1]
    ### Centre location of the filter for which value is calculated
    if kernel_size % 2 == 1:
        centre_location = upscale_factor - 1
    else:
        centre_location = upscale_factor - 0.5

    bilinear = np.zeros([filter_shape[0], filter_shape[1]])
    for x in range(filter_shape[0]):
        for y in range(filter_shape[1]):
            ##Interpolation Calculation
            value = (1 - abs((x - centre_location) / upscale_factor)) * (
            1 - abs((y - centre_location) / upscale_factor))
            bilinear[x, y] = value
    weights = np.zeros(filter_shape)
    for i in range(filter_shape[2]):
        weights[:, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)


    bilinear_weights = tf.get_variable(name="decon_bilinear_filter", initializer=init,
                                       shape=weights.shape)
    return bilinear_weights


KEEP_RATE = 0.8
KEEP_PROB = tf.placeholder(tf.float32)


def FCN_model(data):
    """
     data must be a 4D tensor generated by tf.constant()
      applied on a 4D numpy array with [image index, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNEL]
    """

    # Could use tf.truncated_normal() instead of tf.random_normal(),
    # see what is the best choice
    # note tf.truncated() is used in the FCN implementation
    weights = {'W_conv1': tf.Variable(tf.truncated_normal([3, 3, NUM_CHANNELS, 64])),
               'W_conv2': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
               # Number of pools done until now: 2 -> image size is (IMG_WIDTH / 4 * IMG_HEIGHT / 4)
               'W_fc': tf.Variable(tf.truncated_normal([int(IMG_WIDTH / 4 * IMG_HEIGHT / 4) * 64,
                                                        1024])),
               # 7*7*64 is the dimension of the lower layer after reshaping (28 by 28 images)
               'out': tf.Variable(tf.truncated_normal([1024, NUM_CLASSES]))}

    biases = {'B_conv1': tf.Variable(tf.truncated_normal([64])),
               'B_conv2': tf.Variable(tf.truncated_normal([64])),
               'B_fc': tf.Variable(tf.truncated_normal([1024])),
              # 7*7*64 is the dimension of the lower layer after reshaping (28 by 28 images)
               'out': tf.Variable(tf.truncated_normal([NUM_CLASSES]))}

    # IMPORTANT STEP
    # From FCN implementation:
    # shape = tf.shape(data)
    # deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
    data = tf.reshape(data, shape=[BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS])

    conv1 = conv2d(data, weights['W_conv1'])
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, biases['B_conv1']))

    bn_conv_relu_1 = bn_conv_relu(relu1)
    bn_conv_relu_2 = bn_conv_relu(bn_conv_relu_1)
    pool1 = maxpool2d(bn_conv_relu_2)

    bn_conv_relu_3 = bn_conv_relu(pool1)
    bn_conv_relu_4 = bn_conv_relu(bn_conv_relu_3)
    bn_upconv_relu1 = bn_upconv_relu(bn_conv_relu_4)

    concat1 = tf.concat([bn_conv_relu_1, bn_upconv_relu1], axis=3)
    bn_conv_relu_5 = bn_conv_relu(concat1, 128, 96)
    bn_conv_relu_6 = bn_conv_relu(bn_conv_relu_5, 96, 64)
    convout1 = conv2d(bn_conv_relu_6, tf.Variable(tf.truncated_normal([1, 1, 64, 1])))
    output = tf.nn.sigmoid(convout1, name="output")



    # fc = tf.reshape(pool2, [-1, int(IMG_WIDTH / 4 * IMG_HEIGHT / 4) * 64])
    # fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'] + biases['B_fc']))
    #
    # dropout = tf.nn.dropout(fc, KEEP_RATE)
    #
    # output = tf.matmul(dropout, weights['out'] + biases['out'])
    MODEL = OUTPUT
    return output


# def shuffle_training_set(data, labels):
#
#
#
# def next_batch(data, labels, batch_size):



def train_neural_network(x):
    # the prediction variable is often called logits in tensor flow vocabulary

    prediction = FCN_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    print("train data shape: ", data.shape)
    print("labels shape: ", labels.shape)

    hm_epochs = NUM_EPOCHS
    image_indices = np.array(range(data.shape[0]))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            shuffle(image_indices)
            count_batch = 0
            epoch_loss = 0
            for _ in range(int(TRAIN_SIZE/BATCH_SIZE)):
                # feeding epoch_x and epoch_y for training current batch (to replace with our own )
                epoch_x = data[[image_indices[count_batch:count_batch+BATCH_SIZE]]]
                epoch_y = labels[[image_indices[count_batch:count_batch + BATCH_SIZE]]]
                count_batch += BATCH_SIZE
                print(epoch_x.shape)
                print(epoch_y.shape)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))


        output = sess.run(tf.constant(data[:1]))

        Image.fromarray(output[0]).convert("RGBA").save("result.png")
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x:data[:1], y:labels[:1]}))




def main():
    train_neural_network(x)


if __name__ == '__main__':
    main()
