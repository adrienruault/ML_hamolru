import os
import numpy as np
import tensorflow as tf
import utilities as utils
import utilities_image as utils_img
from random import shuffle

# To be changed to 2
NUM_CLASSES = 2

# To be changed to 400, 400
IMG_WIDTH = 400
IMG_HEIGHT = 400

NUM_EPOCHS = 5

# To be changed to 3
NUM_CHANNELS = 3

# To be changed to 1
BATCH_SIZE = 2

TRAIN_SIZE = 20
TEST_SIZE = 10

REGUL_PARAM = 1e-8

# Set to default
LEARNING_RATE = 0.01

DROPOUT = 0.8

OUTPUT_PATH = './conv_net_output/'
MODEL_PATH = OUTPUT_PATH + 'conv_net_model/conv_net_model.ckpt'
TRAINING_PATH = '../data/training/'

PREDICTION_PATH = './conv_net_prediction/'




def conv_net_model(x, keep_prob):
    """
     data must be a 4D tensor generated by tf.constant()
      applied on a 4D numpy array with [image index, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNEL]
    """

    # Could use tf.truncated_normal() instead of tf.random_normal(),
    # see what is the best choice
    # note tf.truncated() is used in the FCN implementation
    with tf.variable_scope('weights'):
        weights = {'W_conv1': tf.Variable(tf.truncated_normal([3, 3, NUM_CHANNELS, 64]), name = 'W_conv1'),
                   'W_conv2': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name = 'W_conv2'),
                   'W_conv3': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name = 'W_conv3'),
                   'W_conv4': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name = 'W_conv4'),
                   'W_conv5': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name = 'W_conv5'),
                   'W_conv6': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name = 'W_conv6'),
                   'W_deconv1': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name = 'W_deconv1'),
                   'W_conv7': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name = 'W_conv7'),
                   'W_conv8': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name = 'W_conv8'),
                   'W_deconv2': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name = 'W_deconv2'),
                   'W_deconv3': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name = 'W_deconv3'),
                   'W_conv9': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name = 'W_conv9'),
                   'W_conv10': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name = 'W_conv10'),
                   'W_conv11': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name = 'W_conv11'),
                   'W_conv12': tf.Variable(tf.truncated_normal([3, 3, 128, 96]), name = 'W_conv12'),
                   'W_conv13': tf.Variable(tf.truncated_normal([3, 3, 96, 64]), name = 'W_conv13'),
                   'W_conv14': tf.Variable(tf.truncated_normal([3, 3, 128, 96]), name = 'W_conv14'),
                   'W_conv15': tf.Variable(tf.truncated_normal([3, 3, 96, 64]), name = 'W_conv15'),
                   'W_conv16': tf.Variable(tf.truncated_normal([3, 3, 128, 96]), name = 'W_conv16'),
                   'W_conv17': tf.Variable(tf.truncated_normal([3, 3, 96, 64]), name = 'W_conv17'),
                   'W_convout': tf.Variable(tf.truncated_normal([1, 1, 64, NUM_CLASSES]), name = 'W_convout'),
                   'W_conv_supp1': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name='W_conv_supp1'),
                   'W_conv_supp2': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name='W_conv_supp2'),
                   'W_conv_supp3': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name='W_conv_supp3'),
                   'W_conv_supp4': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name='W_conv_supp4'),
                   'W_conv_supp5': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name='W_conv_supp5'),
                   'W_conv_supp6': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name='W_conv_supp6'),
                   'W_conv_supp7': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name='W_conv_supp7'),
                   'W_conv_supp8': tf.Variable(tf.truncated_normal([3, 3, 128, 96]), name='W_conv_supp8'),
                   'W_conv_supp9': tf.Variable(tf.truncated_normal([3, 3, 96, 64]), name='W_conv_supp9'),
                   'W_conv_supp10': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name='W_conv_supp10'),
                   'W_conv_supp11': tf.Variable(tf.truncated_normal([3, 3, 128, 96]), name='W_conv_supp11'),
                   'W_conv_supp12': tf.Variable(tf.truncated_normal([3, 3, 96, 64]), name='W_conv_supp12'),
                   'W_conv_supp13': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name='W_conv_supp13'),
                   }

    with tf.variable_scope('biases'):
        biases = {'B_conv1': tf.Variable(tf.truncated_normal([64]), name = 'B_conv1'),
                   'B_conv2': tf.Variable(tf.truncated_normal([64]), name = 'B_conv2'),
                   'B_conv3': tf.Variable(tf.truncated_normal([64]), name = 'B_conv3'),
                   'B_conv4': tf.Variable(tf.truncated_normal([64]), name = 'B_conv4'),
                   'B_conv5': tf.Variable(tf.truncated_normal([64]), name = 'B_conv5'),
                   'B_conv6': tf.Variable(tf.truncated_normal([64]), name = 'B_conv6'),
                   'B_deconv1': tf.Variable(tf.truncated_normal([64]), name = 'B_deconv1'),
                   'B_conv7': tf.Variable(tf.truncated_normal([64]), name = 'B_conv7'),
                   'B_conv8': tf.Variable(tf.truncated_normal([64]), name = 'B_conv8'),
                   'B_deconv2': tf.Variable(tf.truncated_normal([64]), name = 'B_deconv2'),
                   'B_deconv3': tf.Variable(tf.truncated_normal([64]), name = 'B_deconv3'),
                   'B_conv9': tf.Variable(tf.truncated_normal([64]), name = 'B_conv9'),
                   'B_conv10': tf.Variable(tf.truncated_normal([64]), name = 'B_conv10'),
                   'B_conv11': tf.Variable(tf.truncated_normal([64]), name = 'B_conv11'),
                   'B_conv12': tf.Variable(tf.truncated_normal([96]), name = 'B_conv12'),
                   'B_conv13': tf.Variable(tf.truncated_normal([64]), name = 'B_conv13'),
                   'B_conv14': tf.Variable(tf.truncated_normal([96]), name = 'B_conv14'),
                   'B_conv15': tf.Variable(tf.truncated_normal([64]), name = 'B_conv15'),
                   'B_conv16': tf.Variable(tf.truncated_normal([96]), name = 'B_conv16'),
                   'B_conv17': tf.Variable(tf.truncated_normal([64]), name = 'B_conv17'),
                   'B_convout': tf.Variable(tf.truncated_normal([NUM_CLASSES]), name = 'B_convout'),
                   'B_conv_supp1': tf.Variable(tf.truncated_normal([64]), name='B_conv1_supp1'),
                   'B_conv_supp2': tf.Variable(tf.truncated_normal([64]), name='B_conv1_supp2'),
                   'B_conv_supp3': tf.Variable(tf.truncated_normal([64]), name='B_conv1_supp3'),
                   'B_conv_supp4': tf.Variable(tf.truncated_normal([64]), name='B_conv1_supp4'),
                   'B_conv_supp5': tf.Variable(tf.truncated_normal([64]), name='B_conv1_supp5'),
                   'B_conv_supp6': tf.Variable(tf.truncated_normal([64]), name='B_conv1_supp6'),
                   'B_conv_supp7': tf.Variable(tf.truncated_normal([64]), name='B_conv1_supp7'),
                   'B_conv_supp8': tf.Variable(tf.truncated_normal([96]), name='B_conv1_supp8'),
                   'B_conv_supp9': tf.Variable(tf.truncated_normal([64]), name='B_conv1_supp9'),
                   'B_conv_supp10': tf.Variable(tf.truncated_normal([64]), name='B_conv1_supp10'),
                   'B_conv_supp11': tf.Variable(tf.truncated_normal([96]), name='B_conv1_supp11'),
                   'B_conv_supp12': tf.Variable(tf.truncated_normal([64]), name='B_conv1_supp12'),
                   'B_conv_supp13': tf.Variable(tf.truncated_normal([64]), name='B_conv1_supp13'),
                  }

    with tf.variable_scope('batch_normalization_param'):
        betas = {'beta1': tf.Variable(tf.constant(0.0, shape=[64]), name='beta1', trainable=True),
                 'beta2': tf.Variable(tf.constant(0.0, shape=[64]), name='beta2', trainable=True),
                 'beta3': tf.Variable(tf.constant(0.0, shape=[64]), name='beta3', trainable=True),
                 'beta4': tf.Variable(tf.constant(0.0, shape=[64]), name='beta4', trainable=True),
                 'beta5': tf.Variable(tf.constant(0.0, shape=[64]), name='beta5', trainable=True),
                 'beta6': tf.Variable(tf.constant(0.0, shape=[64]), name='beta6', trainable=True),
                 'beta7': tf.Variable(tf.constant(0.0, shape=[64]), name='beta7', trainable=True),
                 'beta8': tf.Variable(tf.constant(0.0, shape=[64]), name='beta8', trainable=True),
                 'beta9': tf.Variable(tf.constant(0.0, shape=[64]), name='beta9', trainable=True),
                 'beta10': tf.Variable(tf.constant(0.0, shape=[64]), name='beta10', trainable=True),
                 'beta11': tf.Variable(tf.constant(0.0, shape=[128]), name='beta11', trainable=True),
                 'beta12': tf.Variable(tf.constant(0.0, shape=[96]), name='beta12', trainable=True),
                 'beta13': tf.Variable(tf.constant(0.0, shape=[128]), name='beta13', trainable=True),
                 'beta14': tf.Variable(tf.constant(0.0, shape=[96]), name='beta14', trainable=True),
                 'beta15': tf.Variable(tf.constant(0.0, shape=[128]), name='beta15', trainable=True),
                 'beta16': tf.Variable(tf.constant(0.0, shape=[96]), name='beta16', trainable=True),
                 'beta17': tf.Variable(tf.constant(0.0, shape=[64]), name='beta17', trainable=True),
                 'beta18': tf.Variable(tf.constant(0.0, shape=[64]), name='beta18', trainable=True),
                 'beta19': tf.Variable(tf.constant(0.0, shape=[64]), name='beta19', trainable=True),
                 'beta_supp1': tf.Variable(tf.constant(0.0, shape=[64]), name='beta_supp1', trainable=True),
                 'beta_supp2': tf.Variable(tf.constant(0.0, shape=[64]), name='beta_supp2', trainable=True),
                 'beta_supp3': tf.Variable(tf.constant(0.0, shape=[64]), name='beta_supp3', trainable=True),
                 'beta_supp4': tf.Variable(tf.constant(0.0, shape=[64]), name='beta_supp4', trainable=True),
                 'beta_supp5': tf.Variable(tf.constant(0.0, shape=[64]), name='beta_supp5', trainable=True),
                 'beta_supp6': tf.Variable(tf.constant(0.0, shape=[64]), name='beta_supp6', trainable=True),
                 'beta_supp7': tf.Variable(tf.constant(0.0, shape=[64]), name='beta_supp7', trainable=True),
                 'beta_supp8': tf.Variable(tf.constant(0.0, shape=[128]), name='beta_supp8', trainable=True),
                 'beta_supp9': tf.Variable(tf.constant(0.0, shape=[96]), name='beta_supp9', trainable=True),
                 'beta_supp10': tf.Variable(tf.constant(0.0, shape=[64]), name='beta_supp10', trainable=True),
                 'beta_supp11': tf.Variable(tf.constant(0.0, shape=[128]), name='beta_supp11', trainable=True),
                 'beta_supp12': tf.Variable(tf.constant(0.0, shape=[96]), name='beta_supp12', trainable=True),
                 'beta_supp13': tf.Variable(tf.constant(0.0, shape=[64]), name='beta_supp13', trainable=True),
                 }

        gammas = {'gamma1': tf.Variable(tf.constant(1.0, shape=[64]), name='gamma1', trainable=True),
                  'gamma2': tf.Variable(tf.constant(1.0, shape=[64]), name='gamma2', trainable=True),
                  'gamma3': tf.Variable(tf.constant(1.0, shape=[64]), name='gamma3', trainable=True),
                  'gamma4': tf.Variable(tf.constant(1.0, shape=[64]), name='gamma4', trainable=True),
                  'gamma5': tf.Variable(tf.constant(1.0, shape=[64]), name='gamma5', trainable=True),
                  'gamma6': tf.Variable(tf.constant(1.0, shape=[64]), name='gamma6', trainable=True),
                  'gamma7': tf.Variable(tf.constant(1.0, shape=[64]), name='gamma7', trainable=True),
                  'gamma8': tf.Variable(tf.constant(1.0, shape=[64]), name='gamma8', trainable=True),
                  'gamma9': tf.Variable(tf.constant(1.0, shape=[64]), name='gamma9', trainable=True),
                  'gamma10': tf.Variable(tf.constant(1.0, shape=[64]), name='gamma10', trainable=True),
                  'gamma11': tf.Variable(tf.constant(1.0, shape=[128]), name='gamma11', trainable=True),
                  'gamma12': tf.Variable(tf.constant(1.0, shape=[96]), name='gamma12', trainable=True),
                  'gamma13': tf.Variable(tf.constant(1.0, shape=[128]), name='gamma13', trainable=True),
                  'gamma14': tf.Variable(tf.constant(1.0, shape=[96]), name='gamma14', trainable=True),
                  'gamma15': tf.Variable(tf.constant(1.0, shape=[128]), name='gamma15', trainable=True),
                  'gamma16': tf.Variable(tf.constant(1.0, shape=[96]), name='gamma16', trainable=True),
                  'gamma17': tf.Variable(tf.constant(1.0, shape=[64]), name='gamma17', trainable=True),
                  'gamma18': tf.Variable(tf.constant(1.0, shape=[64]), name='gamma18', trainable=True),
                  'gamma19': tf.Variable(tf.constant(1.0, shape=[64]), name='gamma19', trainable=True),
                  'gamma_supp1': tf.Variable(tf.constant(1.0, shape=[64]), name='gamma_supp1', trainable=True),
                  'gamma_supp2': tf.Variable(tf.constant(1.0, shape=[64]), name='gamma_supp2', trainable=True),
                  'gamma_supp3': tf.Variable(tf.constant(1.0, shape=[64]), name='gamma_supp3', trainable=True),
                  'gamma_supp4': tf.Variable(tf.constant(1.0, shape=[64]), name='gamma_supp4', trainable=True),
                  'gamma_supp5': tf.Variable(tf.constant(1.0, shape=[64]), name='gamma_supp5', trainable=True),
                  'gamma_supp6': tf.Variable(tf.constant(1.0, shape=[64]), name='gamma_supp6', trainable=True),
                  'gamma_supp7': tf.Variable(tf.constant(1.0, shape=[64]), name='gamma_supp7', trainable=True),
                  'gamma_supp8': tf.Variable(tf.constant(1.0, shape=[128]), name='gamma_supp8', trainable=True),
                  'gamma_supp9': tf.Variable(tf.constant(1.0, shape=[96]), name='gamma_supp9', trainable=True),
                  'gamma_supp10': tf.Variable(tf.constant(1.0, shape=[64]), name='gamma_supp10', trainable=True),
                  'gamma_supp11': tf.Variable(tf.constant(1.0, shape=[128]), name='gamma_supp11', trainable=True),
                  'gamma_supp12': tf.Variable(tf.constant(1.0, shape=[96]), name='gamma_supp12', trainable=True),
                  'gamma_supp13': tf.Variable(tf.constant(1.0, shape=[64]), name='gamma_supp13', trainable=True),
                  }

    # IMPORTANT STEP
    # From FCN implementation:
    # shape = tf.shape(data)
    # deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
    #data = tf.reshape(data, shape=[BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS])

    # Going down
    conv_relu1 = utils.conv2d_relu(x, weights['W_conv1'], biases['B_conv1'], name='conv_relu1')

    bn_conv_relu1 = utils.bn_conv_relu(conv_relu1, weights['W_conv2'], biases['B_conv2'],
                                       betas['beta1'], gammas['gamma1'], name='bn_conv_relu1')

    bn_conv_relu2 = utils.bn_conv_relu(bn_conv_relu1, weights['W_conv3'], biases['B_conv3'],
                                       betas['beta2'], gammas['gamma2'], name='bn_conv_relu2')

    max_pool1 = utils.maxpool2d(bn_conv_relu2, name='max_pool_1')

    bn_conv_relu3 = utils.bn_conv_relu(max_pool1, weights['W_conv4'], biases['B_conv4'],
                                       betas['beta3'], gammas['gamma3'], name='bn_conv_relu3')

    bn_conv_relu4 = utils.bn_conv_relu(bn_conv_relu3, weights['W_conv5'], biases['B_conv5'],
                                       betas['beta4'], gammas['gamma4'], name='bn_conv_relu4')

    bn_conv_relu5 = utils.bn_conv_relu(bn_conv_relu4, weights['W_conv6'], biases['B_conv6'],
                                      betas['beta5'], gammas['gamma5'], name='bn_conv_relu4')

    max_pool2 = utils.maxpool2d(bn_conv_relu5, name='max_pool2')

    bn_conv_relu6 = utils.bn_conv_relu(max_pool2, weights['W_conv7'], biases['B_conv7'],
                                      betas['beta6'], gammas['gamma6'], name='bn_conv_relu6')

    bn_conv_relu7 = utils.bn_conv_relu(bn_conv_relu6, weights['W_conv8'], biases['B_conv8'],
                                      betas['beta7'], gammas['gamma7'], name='bn_conv_relu7')

    bn_conv_relu8 = utils.bn_conv_relu(bn_conv_relu7, weights['W_conv9'], biases['B_conv9'],
                                      betas['beta8'], gammas['gamma8'], name='bn_conv_relu8')

    max_pool3 = utils.maxpool2d(bn_conv_relu8, name='max_pool3')

    bn_conv_relu9 = utils.bn_conv_relu(max_pool3, weights['W_conv10'], biases['B_conv10'],
                                      betas['beta9'], gammas['gamma9'], name='bn_conv_relu9')

    bn_conv_relu10 = utils.bn_conv_relu(bn_conv_relu9, weights['W_conv11'], biases['B_conv11'],
                                      betas['beta10'], gammas['gamma10'], name='bn_conv_relu10')

    # Enlarge model, going to 25x25 from 100x100

    bn_conv_relu_supp1 = utils.bn_conv_relu(bn_conv_relu9, weights['W_conv_supp1'], biases['B_conv_supp1'],
                                        betas['beta_supp1'], gammas['gamma_supp1'], name='bn_conv_supp1')

    max_pool_supp1 = utils.maxpool2d(bn_conv_relu_supp1, name='max_pool_supp1')

    bn_conv_relu_supp2 = utils.bn_conv_relu(max_pool_supp1, weights['W_conv_supp2'], biases['B_conv_supp2'],
                                            betas['beta_supp2'], gammas['gamma_supp2'], name='bn_conv_supp2')

    bn_conv_relu_supp3 = utils.bn_conv_relu(bn_conv_relu_supp2, weights['W_conv_supp3'], biases['B_conv_supp3'],
                                            betas['beta_supp3'], gammas['gamma_supp3'], name='bn_conv_supp3')

    bn_conv_relu_supp4 = utils.bn_conv_relu(bn_conv_relu_supp3, weights['W_conv_supp4'], biases['B_conv_supp4'],
                                            betas['beta_supp4'], gammas['gamma_supp4'], name='bn_conv_supp4')

    max_pool_supp2 = utils.maxpool2d(bn_conv_relu_supp4, name='max_pool_supp2')

    bn_conv_relu_supp5 = utils.bn_conv_relu(max_pool_supp2, weights['W_conv_supp5'], biases['B_conv_supp5'],
                                            betas['beta_supp5'], gammas['gamma_supp5'], name='bn_conv_supp5')

    bn_conv_relu_supp6 = utils.bn_conv_relu(bn_conv_relu_supp5, weights['W_conv_supp6'], biases['B_conv_supp6'],
                                            betas['beta_supp6'], gammas['gamma_supp6'], name='bn_conv_supp6')

    bn_deconv_relu_supp1 = utils.bn_deconv_relu(bn_conv_relu_supp6, weights['W_conv_supp7'],
                                                biases['B_conv_supp7'],
                                           betas['beta_supp7'], gammas['gamma_supp7'],
                                           upscale_factor=2, name='bn_deconv_relu_supp1')

    concat_supp1 = tf.concat([bn_conv_relu_supp3, bn_deconv_relu_supp1], axis=3, name='concat2')

    bn_conv_relu_supp8 = utils.bn_conv_relu(concat_supp1, weights['W_conv_supp8'], biases['B_conv_supp8'],
                                            betas['beta_supp8'], gammas['gamma_supp8'], name='bn_conv_supp8')

    bn_conv_relu_supp9 = utils.bn_conv_relu(bn_conv_relu_supp8, weights['W_conv_supp9'], biases['B_conv_supp9'],
                                            betas['beta_supp9'], gammas['gamma_supp9'], name='bn_conv_supp9')

    bn_deconv_relu_supp2 = utils.bn_deconv_relu(bn_conv_relu_supp9, weights['W_conv_supp10'],
                                                biases['B_conv_supp10'],
                                                betas['beta_supp10'], gammas['gamma_supp10'],
                                                upscale_factor=2, name='bn_deconv_relu_supp2')

    concat_supp2 = tf.concat([bn_conv_relu10, bn_deconv_relu_supp2], axis=3, name='concat2')

    bn_conv_relu_supp11 = utils.bn_conv_relu(concat_supp2, weights['W_conv_supp11'], biases['B_conv_supp11'],
                                            betas['beta_supp11'], gammas['gamma_supp11'], name='bn_conv_supp11')

    bn_conv_relu_supp12 = utils.bn_conv_relu(bn_conv_relu_supp11, weights['W_conv_supp12'],
                                             biases['B_conv_supp12'],
                                            betas['beta_supp12'], gammas['gamma_supp12'], name='bn_conv_supp12')

    # Going up
    bn_deconv_relu1 = utils.bn_deconv_relu(bn_conv_relu_supp12, weights['W_deconv1'], biases['B_deconv1'],
                                           betas['beta17'],gammas['gamma17'],
                                           upscale_factor=2, name='bn_deconv_relu1')

    concat1 = tf.concat([bn_conv_relu7, bn_deconv_relu1], axis=3, name='concat1')

    bn_conv_relu11 = utils.bn_conv_relu(concat1, weights['W_conv12'], biases['B_conv12'],
                                      betas['beta11'], gammas['gamma11'], name='bn_conv_relu11')

    bn_conv_relu12 = utils.bn_conv_relu(bn_conv_relu11, weights['W_conv13'], biases['B_conv13'],
                                       betas['beta12'], gammas['gamma12'], name='bn_conv_relu12')

    bn_deconv_relu2 = utils.bn_deconv_relu(bn_conv_relu12, weights['W_deconv2'], biases['B_deconv2'],
                                           betas['beta18'], gammas['gamma18'],
                                       upscale_factor=2, name='bn_deconv_relu2')

    concat2 = tf.concat([bn_conv_relu4, bn_deconv_relu2], axis=3, name='concat2')

    bn_conv_relu13 = utils.bn_conv_relu(concat2, weights['W_conv14'], biases['B_conv14'],
                                       betas['beta13'], gammas['gamma13'], name='bn_conv_relu13')

    bn_conv_relu14 = utils.bn_conv_relu(bn_conv_relu13, weights['W_conv15'], biases['B_conv15'],
                                       betas['beta14'], gammas['gamma14'], name='bn_conv_relu14')

    bn_deconv_relu3 = utils.bn_deconv_relu(bn_conv_relu14, weights['W_deconv3'], biases['B_deconv3'],
                                           betas['beta19'], gammas['gamma19'],
                                           upscale_factor=2, name='bn_deconv_relu3')

    concat3 = tf.concat([bn_conv_relu1, bn_deconv_relu3], axis=3, name='concat3')

    bn_conv_relu15 = utils.bn_conv_relu(concat3, weights['W_conv16'], biases['B_conv16'],
                                       betas['beta15'], gammas['gamma15'], name='bn_conv_relu15')

    bn_conv_relu16 = utils.bn_conv_relu(bn_conv_relu15, weights['W_conv17'], biases['B_conv17'],
                                       betas['beta16'], gammas['gamma16'], name='bn_conv_relu16')

    convout = utils.conv2d_relu(bn_conv_relu16, weights['W_convout'], biases['B_convout'], name='convout')


    # Storing variables into a variables list
    # weights:
    variables = []
    for _, item in weights.items():
        variables.append(item)

    for _, item in biases.items():
        variables.append(item)


    return convout, variables




class ConvNet(object):

    def __init__(self):
        self.x = tf.placeholder(tf.float32, shape = [None, IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS])
        self.y = tf.placeholder(tf.float32, shape = [None, IMG_WIDTH, IMG_HEIGHT, NUM_CLASSES])
        self.keep_prob = tf.placeholder(tf.float32)

        logits, self.variables = conv_net_model(self.x, self.keep_prob)

        self.cost = self._get_cost(logits)

        self.gradients_node = tf.gradients(self.cost, self.variables)

        # Only computed for the sake of summary in TensorBoard
        self.cross_entropy = tf.reduce_mean(utils.cross_entropy(tf.reshape(self.y, [-1, NUM_CLASSES]),
                                tf.reshape(utils.pixel_wise_softmax_2(logits), [-1, NUM_CLASSES])))

        self.predicter = utils.pixel_wise_softmax_2(logits)

        self.correct_pred = tf.equal(tf.argmax(self.predicter, 3), tf.argmax(self.y, 3))

        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def _get_cost(self, logits):
        flat_logits = tf.reshape(logits, [-1, NUM_CLASSES])
        flat_labels = tf.reshape(self.y, [-1, NUM_CLASSES])

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                        labels=flat_labels))
        # add regularization
        regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
        loss += REGUL_PARAM * regularizers

        return loss

    def predict(self, x_test):
        init = tf.global_variables_initializer()

        with tf.Session as sess:
            sess.run(init)

            # Restoring the model from previous train
            # We take the model from the path specified in MODEL_PATH
            self.restore(sess, MODEL_PATH)

            y_useless = np.empty((BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, NUM_CLASSES))
            prediction = sess.run(self.predicter, feed_dict={self.x : x_test, self.y : y_useless,
                                                             self.keep_prob : 1.})

        return prediction

    def save(self, sess, model_save_path):
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_save_path)

        return save_path

    def restore(self, sess):
        saver = tf.train.Saver()
        saver.restore(sess, MODEL_PATH)



class Trainer(object):

    def __init__(self, conv_net):
        self.conv_net = conv_net

    def _get_optimizer(self, global_step):
        self.learning_rate_node = tf.Variable(LEARNING_RATE)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node)\
                                            .minimize(self.conv_net.cost, global_step=global_step)
        return optimizer

    def _initialize(self):
        global_step = tf.Variable(0)

        self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.conv_net.gradients_node)]))

        # Definition of the summaries
        tf.summary.histogram('norm_grads', self.norm_gradients_node)
        tf.summary.scalar('loss', self.conv_net.cost)
        tf.summary.scalar('cross_entropy', self.conv_net.cross_entropy)
        tf.summary.scalar('accuracy', self.conv_net.accuracy)

        self.optimizer = self._get_optimizer(global_step)
        tf.summary.scalar('learning_rate', self.learning_rate_node)

        # Merging all summaries for tensor flow
        self.summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()

        return init


    def train(self, restore=False):

        if not os.path.exists(OUTPUT_PATH + 'train_predictions/'):
            os.makedirs(OUTPUT_PATH + 'train_predictions/')

        print('-> Loading data:')
        data = utils_img.load_images(TRAINING_PATH + "images/", TRAIN_SIZE)
        labels = utils_img.load_groundtruths(TRAINING_PATH + "groundtruth/", TRAIN_SIZE)

        print("train data shape: ", data.shape)
        print("labels shape: ", labels.shape)


        init = self._initialize()

        with tf.Session() as sess:
            # writing the graph
            tf.train.write_graph(sess.graph_def, OUTPUT_PATH, 'graph.pb', False)

            sess.run(init)

            if restore:
                ckpt = tf.train.get_checkpoint_state(OUTPUT_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    self.conv_net.restore(sess, ckpt.model_checkpoint_path)

            summary_writer = tf.summary.FileWriter(OUTPUT_PATH + 'summary/', graph=sess.graph)


            hm_epochs = NUM_EPOCHS
            image_indices = np.array(range(data.shape[0]))
            for epoch in range(hm_epochs):
                shuffle(image_indices)
                count_batch = 0
                epoch_loss = 0
                for step in range(int(TRAIN_SIZE/BATCH_SIZE)):
                    # feeding epoch_x and epoch_y for training current batch (to replace with our own )
                    epoch_x = data[[image_indices[count_batch:count_batch+BATCH_SIZE]]]
                    epoch_y = labels[[image_indices[count_batch:count_batch + BATCH_SIZE]]]
                    count_batch += BATCH_SIZE
                    img_idx = [image_indices[count_batch:count_batch+BATCH_SIZE]][0]

                    # sees.run() evaluate a tensor
                    # the first argument of sess.run() is an array corresponding to every operation needed
                    # Feeding sess.run() with y is necessary because cost_op needs it
                    _, loss, lr, gradients = sess.run((self.optimizer,
                                                        self.conv_net.cost, self.learning_rate_node,
                                                       self.conv_net.gradients_node),
                                                        feed_dict={self.conv_net.x: epoch_x,
                                                                   self.conv_net.y: epoch_y,
                                                                    self.conv_net.keep_prob: DROPOUT})
                    epoch_loss += loss
                    print('Step', step+1, 'completed out of', int(TRAIN_SIZE/BATCH_SIZE), '/ step loss:', loss)

                print('epoch_x shape:', epoch_x.shape)
                print('epoch_y shape:', epoch_y.shape)


                print('-> Epoch', epoch+1, 'completed out of', hm_epochs, '/ epoch loss:', epoch_loss)
                print('Current batch size:', BATCH_SIZE)

                # In the future convert it in test set
                for i in range(TRAIN_SIZE):
                    img = data[[i]]
                    groundtruth = labels[[i]]
                    accuracy = self.output_stats(sess, summary_writer, step, img, groundtruth)
                    print('Image' + str(i+1) + ' accuracy:', accuracy, '%%')
                    self.store_prediction(sess, img, groundtruth,
                                            save_path = OUTPUT_PATH +\
                                                        "train_predictions/image_{a}_epoch_{b}.png".format(a=(i+1),
                                                                                                           b=epoch))

                model_path = self.conv_net.save(sess, MODEL_PATH)
            return model_path


    def output_stats(self, sess, summary_writer, step, batch_x, batch_y):
        # Calculate batch loss and accuracy
        summary_str, loss, acc, predictions = sess.run([self.summary_op, self.conv_net.cost,
                                                        self.conv_net.accuracy, self.conv_net.predicter],
                                                        feed_dict={self.conv_net.x: batch_x,
                                                                   self.conv_net.y: batch_y,
                                                                   self.conv_net.keep_prob: 1.})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        return utils.accuracy(predictions, batch_y)



    def store_prediction(self, sess, test_x, test_y, save_path):
        predictions = sess.run(self.conv_net.predicter, feed_dict={self.conv_net.x: test_x,
                                                                   self.conv_net.y: test_y,
                                                                   self.conv_net.keep_prob: 1.})

        utils_img.save_image_from_proba_pred(predictions, save_path)




def main():
    conv_net = ConvNet()
    trainer = Trainer(conv_net)
    save_model_path = trainer.train()
    print('Model saved in:', save_model_path)


if __name__ == '__main__':
    main()
