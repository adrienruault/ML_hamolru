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

NUM_EPOCHS = 20

# To be changed to 3
NUM_CHANNELS = 3

# To be changed to 1
BATCH_SIZE = 2

TRAIN_SIZE = 100
TEST_SIZE = 10

RECORDING_STEP = 5

# initially set to 5e-4 but it is maybe too much
REGUL_PARAM = 1e-15

# Default is 0.001 for AdamOptimizer
LEARNING_RATE = 0.001

DROPOUT = 1.0

STD_VAR_INIT = 0.1

BLOCK_NUMBER = 5

OUTPUT_PATH = './conv_net_output/'
MODEL_PATH = OUTPUT_PATH + 'conv_net_model/conv_net_model.ckpt'
TRAINING_PATH = '../data/training/'

PREDICTION_PATH = './conv_net_prediction/'




def conv_net_model(x, keep_prob, phase_train):
    """
     data must be a 4D tensor generated by tf.constant()
      applied on a 4D numpy array with [image index, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNEL]
    """

    # Could use tf.truncated_normal() instead of tf.random_normal(),
    # see what is the best choice
    # note tf.truncated() is used in the FCN implementation
    with tf.variable_scope('weights'):
        weights = {'W_d_conv1': utils.weight_def([3, 3, NUM_CHANNELS, 64], stddev=STD_VAR_INIT, name = 'W_d_conv1'),
                   'W_d_conv2': utils.weight_def([3, 3, 64, 64], stddev=STD_VAR_INIT, name = 'W_d_conv2'),
                   'W_d_conv3': utils.weight_def([3, 3, 64, 64], stddev=STD_VAR_INIT, name = 'W_d_conv3'),
                   'W_d_conv4': utils.weight_def([3, 3, 64, 64], stddev=STD_VAR_INIT, name = 'W_d_conv4'),
                   'W_d_conv5': utils.weight_def([3, 3, 64, 64], stddev=STD_VAR_INIT, name = 'W_d_conv5'),
                   'W_d_conv6': utils.weight_def([3, 3, 64, 64], stddev=STD_VAR_INIT, name = 'W_d_conv6'),
                   'W_d_conv7': utils.weight_def([3, 3, 64, 64], stddev=STD_VAR_INIT, name = 'W_d_conv7'),
                   'W_d_conv8': utils.weight_def([3, 3, 64, 64], stddev=STD_VAR_INIT, name = 'W_d_conv8'),
                   'W_d_conv9': utils.weight_def([3, 3, 64, 64], stddev=STD_VAR_INIT, name = 'W_d_conv9'),
                   'W_d_conv10': utils.weight_def([3, 3, 64, 64], stddev=STD_VAR_INIT, name = 'W_d_conv10'),
                   #'W_d_conv11': utils.weight_def([3, 3, 64, 64], stddev=STD_VAR_INIT, name = 'W_d_conv11'),
                   #'W_d_conv12': utils.weight_def([3, 3, 64, 64], stddev=STD_VAR_INIT, name = 'W_d_conv12'),

                   'W_trans1': utils.weight_def([3, 3, 64, 64], stddev=STD_VAR_INIT, name = 'W_trans1'),
                   'W_trans2': utils.weight_def([3, 3, 64, 64], stddev=STD_VAR_INIT, name = 'W_trans2'),

                   'W_deconv1': utils.weight_def([3, 3, 64, 64], stddev=STD_VAR_INIT, name = 'W_deconv1'),
                   'W_u_conv1': utils.weight_def([3, 3, 128, 96], stddev=STD_VAR_INIT, name = 'W_u_conv1'),
                   'W_u_conv2': utils.weight_def([3, 3, 96, 64], stddev=STD_VAR_INIT, name = 'W_u_conv2'),

                   'W_deconv2': utils.weight_def([3, 3, 64, 64], stddev=STD_VAR_INIT, name = 'W_deconv2'),
                   'W_u_conv3': utils.weight_def([3, 3, 128, 96], stddev=STD_VAR_INIT, name = 'W_u_conv3'),
                   'W_u_conv4': utils.weight_def([3, 3, 96, 64], stddev=STD_VAR_INIT, name = 'W_u_conv4'),

                   'W_deconv3': utils.weight_def([3, 3, 64, 64], stddev=STD_VAR_INIT, name = 'W_deconv3'),
                   'W_u_conv5': utils.weight_def([3, 3, 128, 96], stddev=STD_VAR_INIT, name = 'W_u_conv5'),
                   'W_u_conv6': utils.weight_def([3, 3, 96, 64], stddev=STD_VAR_INIT, name = 'W_u_conv6'),

                   'W_deconv4': utils.weight_def([3, 3, 64, 64], stddev=STD_VAR_INIT, name = 'W_deconv4'),
                   'W_u_conv7': utils.weight_def([3, 3, 128, 96], stddev=STD_VAR_INIT, name = 'W_u_conv7'),
                   'W_u_conv8': utils.weight_def([3, 3, 96, 64], stddev=STD_VAR_INIT, name = 'W_u_conv8'),

                   'W_deconv5': utils.weight_def([3, 3, 64, 64], stddev=STD_VAR_INIT, name = 'W_deconv5'),
                   'W_u_conv9': utils.weight_def([3, 3, 128, 96], stddev=STD_VAR_INIT, name = 'W_u_conv9'),
                   'W_u_conv10': utils.weight_def([3, 3, 96, 64], stddev=STD_VAR_INIT, name = 'W_u_conv10'),

                   #'W_deconv6': utils.weight_def([3, 3, 64, 64], stddev=STD_VAR_INIT, name = 'W_deconv6'),
                   #'W_u_conv11': utils.weight_def([3, 3, 128, 96], stddev=STD_VAR_INIT, name = 'W_u_conv11'),
                   #'W_u_conv12': utils.weight_def([3, 3, 96, 64], stddev=STD_VAR_INIT, name = 'W_u_conv12'),

                   'W_convout': utils.weight_def([1, 1, 64, NUM_CLASSES], stddev=STD_VAR_INIT, name = 'W_convout')}

    with tf.variable_scope('biases'):
        biases = {'B_d_conv1': utils.bias_def([64], name = 'B_d_conv1'),
                   'B_d_conv2': utils.bias_def([64], name = 'B_d_conv2'),
                   'B_d_conv3': utils.bias_def([64], name = 'B_d_conv3'),
                   'B_d_conv4': utils.bias_def([64], name = 'B_d_conv4'),
                   'B_d_conv5': utils.bias_def([64], name = 'B_d_conv5'),
                   'B_d_conv6': utils.bias_def([64], name = 'B_d_conv6'),
                   'B_d_conv7': utils.bias_def([64], name = 'B_d_conv7'),
                   'B_d_conv8': utils.bias_def([64], name = 'B_d_conv8'),
                   'B_d_conv9': utils.bias_def([64], name = 'B_d_conv9'),
                   'B_d_conv10': utils.bias_def([64], name = 'B_d_conv10'),
                   'B_d_conv11': utils.bias_def([64], name = 'B_d_conv11'),
                   'B_d_conv12': utils.bias_def([64], name = 'B_d_conv12'),

                   'B_trans1': utils.bias_def([64], name = 'B_trans1'),
                   'B_trans2': utils.bias_def([64], name = 'B_trans2'),

                   'B_deconv1': utils.bias_def([64], name = 'B_deconv1'),
                   'B_u_conv1': utils.bias_def([96], name = 'B_u_conv1'),
                   'B_u_conv2': utils.bias_def([64], name = 'B_u_conv2'),

                   'B_deconv2': utils.bias_def([64], name = 'B_deconv2'),
                   'B_u_conv3': utils.bias_def([96], name = 'B_u_conv3'),
                   'B_u_conv4': utils.bias_def([64], name = 'B_u_conv4'),

                   'B_deconv3': utils.bias_def([64], name = 'B_deconv3'),
                   'B_u_conv5': utils.bias_def([96], name = 'B_u_conv5'),
                   'B_u_conv6': utils.bias_def([64], name = 'B_u_conv6'),

                   'B_deconv4': utils.bias_def([64], name = 'B_deconv4'),
                   'B_u_conv7': utils.bias_def([96], name = 'B_u_conv7'),
                   'B_u_conv8': utils.bias_def([64], name = 'B_u_conv8'),

                   'B_deconv5': utils.bias_def([64], name = 'B_deconv5'),
                   'B_u_conv9': utils.bias_def([96], name = 'B_u_conv9'),
                   'B_u_conv10': utils.bias_def([64], name = 'B_u_conv10'),

                   'B_deconv6': utils.bias_def([64], name = 'B_deconv6'),
                   'B_u_conv11': utils.bias_def([96], name = 'B_u_conv11'),
                   'B_u_conv12': utils.bias_def([64], name = 'B_u_conv12'),

                   'B_convout': utils.bias_def([NUM_CLASSES], name = 'B_convout')}

    # IMPORTANT STEP
    # From FCN implementation:
    # shape = tf.shape(data)
    # deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
    #data = tf.reshape(data, shape=[BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS])


    #dep_features = 64


    # Going down
    pool_layers = [x]
    conv_layers = []
    for i in range(BLOCK_NUMBER):
        if i==0:
            in_features = 3
        else:
            in_features = 64

        with tf.name_scope('pool' + str(i+1)):
            d_conv_dropout_relu1 = utils.conv2d_dropout_relu(pool_layers[i],\
                                                            weights['W_d_conv' + str(i*2+1)], biases['B_d_conv' + str(i*2+1)], in_features,\
                                                            keep_prob = keep_prob, phase_train = phase_train, name = 'd_conv_' + str(i*2+1))

            d_conv_dropout_relu2 = utils.conv2d_dropout_relu(d_conv_dropout_relu1,\
                                                            weights['W_d_conv' + str(i*2+2)], biases['B_d_conv' + str(i*2+2)], 64,\
                                                            keep_prob = keep_prob, phase_train = phase_train, name = 'd_conv_' + str(i*2+2))

            max_pool = utils.maxpool2d(d_conv_dropout_relu2, name = 'max_pool' + (str(i+1)))

        conv_layers += [d_conv_dropout_relu2]
        pool_layers += [max_pool]



    # Transition
    with tf.name_scope('transition'):
        d_conv_dropout_relu_trans1 = utils.conv2d_dropout_relu(pool_layers[-1], weights['W_trans1'], biases['B_trans1'], 64, keep_prob = keep_prob,\
                                                            phase_train = phase_train, name='trans1')

        d_conv_dropout_relu_trans2 = utils.conv2d_dropout_relu(d_conv_dropout_relu_trans1,\
                                                            weights['W_trans2'], biases['B_trans2'], 64,\
                                                            keep_prob = keep_prob, phase_train = phase_train, name ='trans2')

    # Going up
    deconv_layers = [d_conv_dropout_relu_trans2]
    for i in range(BLOCK_NUMBER):

        with tf.name_scope('deconv' + str(i+1)):
            deconv_relu1 = utils.deconv2d_relu(deconv_layers[i],\
                                                weights['W_deconv' + str(i+1)], biases['B_deconv' + str(i+1)], 64,\
                                                phase_train = phase_train, name = 'deconv' + str(i+1))

            concat1 = tf.concat([deconv_relu1, conv_layers[BLOCK_NUMBER - (i+1)]], axis = 3, name = 'concat' + str(i+1))

            u_conv_dropout_relu1 = utils.conv2d_dropout_relu(concat1,\
                                                weights['W_u_conv' + str(2*i+1)], biases['B_u_conv' + str(2*i+1)], 128,\
                                                keep_prob = keep_prob, phase_train = phase_train, name ='u_conv_' + str(2*i+1))

            u_conv_dropout_relu2 = utils.conv2d_dropout_relu(u_conv_dropout_relu1,\
                                                weights['W_u_conv' + str(2*i+2)], biases['B_u_conv' + str(2*i+2)], 96,\
                                                keep_prob = keep_prob, phase_train = phase_train, name ='u_conv_' + str(2*i+2))

            deconv_layers += [u_conv_dropout_relu2]


    # Convout
    with tf.name_scope('convout'):
        convout = utils.conv2d_dropout_relu(deconv_layers[-1], weights['W_convout'], biases['B_convout'], 64, keep_prob = tf.constant(1.0),\
                                                            phase_train = phase_train, name = 'convout')

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

        with tf.name_scope('input'):
            self.x = tf.placeholder(tf.float32, shape = [None, IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS])
            self.y = tf.placeholder(tf.float32, shape = [None, IMG_WIDTH, IMG_HEIGHT, NUM_CLASSES])
            self.keep_prob = tf.placeholder(tf.float32)
            self.phase_train = tf.placeholder(tf.bool)

        with tf.name_scope('logits'):
            logits, self.variables = conv_net_model(self.x, self.keep_prob, self.phase_train)

        with tf.name_scope('cost'):
            self.cost = self._get_cost(logits)

        with tf.name_scope('cross_entropy_sum'):
            self.cross_entropy_sum = tf.reduce_sum(\
                                        tf.nn.softmax_cross_entropy_with_logits(\
                                            logits=tf.reshape(logits, [-1, NUM_CLASSES]),
                                            labels=tf.reshape(self.y, [-1, NUM_CLASSES])))

        with tf.name_scope('cross_entropy'):
        # Only computed for the sake of summary in TensorBoard
            self.cross_entropy = tf.reduce_mean(utils.cross_entropy(tf.reshape(self.y, [-1, NUM_CLASSES]),
                                    tf.reshape(utils.pixel_wise_softmax_2(logits), [-1, NUM_CLASSES])))

        with tf.name_scope('softmax_predicter'):
            #self.predicter = utils.pixel_wise_softmax_2(logits)
            self.predicter = tf.nn.softmax(logits, dim = 3)

        with tf.name_scope('accuracy'):
            self.correct_pred = tf.equal(tf.argmax(self.predicter, 3), tf.argmax(self.y, 3))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

    def _get_cost(self, logits):
        flat_logits = tf.reshape(logits, [-1, NUM_CLASSES])
        flat_labels = tf.reshape(self.y, [-1, NUM_CLASSES])

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                        labels=flat_labels))
        # add regularization
        with tf.name_scope('regularizers'):
            regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])

        tf.summary.scalar('regularizers', REGUL_PARAM * regularizers)
        loss += REGUL_PARAM * regularizers

        return loss

    def predict(self, x_test):
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)

            # Restoring the model from previous train
            # We take the model from the path specified in MODEL_PATH
            self.restore(sess, MODEL_PATH)

            y_useless = np.empty((BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, NUM_CLASSES))
            prediction = sess.run(self.predicter, feed_dict={self.x : x_test, self.y : y_useless, self.keep_prob : 1.,\
                                                                self.phase_train: False})

        return prediction

    def save(self, sess, model_save_path):
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_save_path)

        return save_path

    def restore(self, sess, model_save_path):
        saver = tf.train.Saver()
        saver.restore(sess, model_save_path)



class Trainer(object):

    def __init__(self, conv_net):
        self.conv_net = conv_net

    def _get_optimizer(self, global_step):
        self.learning_rate_node = tf.Variable(LEARNING_RATE)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_node)\
                                            .minimize(self.conv_net.cost, global_step=global_step)
        #optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate_node)\
        #                                    .minimize(self.conv_net.cost, global_step=global_step)
        return optimizer

    def _initialize(self):
        global_step = tf.Variable(0)

        # Definition of the summaries
        tf.summary.scalar('loss', self.conv_net.cost)
        tf.summary.scalar('cross_entropy', self.conv_net.cross_entropy)
        tf.summary.scalar('accuracy', self.conv_net.accuracy)

        with tf.name_scope('optimizer'):
            self.optimizer = self._get_optimizer(global_step)

        tf.summary.scalar('learning_rate', self.learning_rate_node)

        #summary_convout = get_image_summary(conv_net.logits)
        #tf.summary.image('convout', summary_convout)

        summary_predicter = utils_img.get_image_summary(self.conv_net.predicter)
        tf.summary.image('convout', summary_predicter)

        summary_argmax = utils_img.get_image_summary(tf.expand_dims(tf.argmax(self.conv_net.predicter, 3), 3))
        tf.summary.image('argmax_convout', summary_argmax)


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
            tf.train.write_graph(sess.graph_def, OUTPUT_PATH + 'summary/', 'graph.pb', False)

            sess.run(init)

            print('trainable:', tf.trainable_variables(scope=None))

            if restore:
                ckpt = tf.train.get_checkpoint_state(OUTPUT_PATH + 'conv_net_model/')
                if ckpt and ckpt.model_checkpoint_path:
                    print('Restoring previous train...')
                    self.conv_net.restore(sess, ckpt.model_checkpoint_path)

            summary_writer = tf.summary.FileWriter(OUTPUT_PATH + 'summary/', graph=sess.graph)


            hm_epochs = NUM_EPOCHS
            image_indices = np.array(range(data.shape[0]))
            for epoch in range(hm_epochs):
                shuffle(image_indices)
                epoch_loss = 0
                for step in range(int(TRAIN_SIZE/BATCH_SIZE)):
                    # feeding epoch_x and epoch_y for training current batch (to replace with our own )
                    epoch_x = data[[image_indices[step:step+BATCH_SIZE]]]
                    epoch_y = labels[[image_indices[step:step + BATCH_SIZE]]]

                    if step % RECORDING_STEP == 0:
                        summary_str, _, loss, lr = sess.run((self.summary_op, self.optimizer, self.conv_net.cost, self.learning_rate_node),\
                                                            feed_dict={self.conv_net.x: epoch_x, self.conv_net.y: epoch_y,
                                                                        self.conv_net.keep_prob: DROPOUT, self.conv_net.phase_train: True})

                        glob_step = epoch *int(TRAIN_SIZE/BATCH_SIZE) + step
                        summary_writer.add_summary(summary_str, glob_step)
                        summary_writer.flush()
                    else:
                        # sees.run() evaluate a tensor
                        # the first argument of sess.run() is an array corresponding to every operation needed
                        # Feeding sess.run() with y is necessary because cost_op needs it
                        _, loss, lr = sess.run((self.optimizer, self.conv_net.cost, self.learning_rate_node),\
                                                            feed_dict={self.conv_net.x: epoch_x, self.conv_net.y: epoch_y,
                                                                        self.conv_net.keep_prob: DROPOUT, self.conv_net.phase_train: True})
                    epoch_loss += loss
                    print('Step', step+1, 'completed out of', int(TRAIN_SIZE/BATCH_SIZE), '/ step loss:', loss)

                print('epoch_x shape:', epoch_x.shape)
                print('epoch_y shape:', epoch_y.shape)


                print('-> Epoch', epoch+1, 'completed out of', hm_epochs, '/ epoch loss:', epoch_loss)
                print('Current batch size:', BATCH_SIZE)

                model_path = self.conv_net.save(sess, MODEL_PATH)

                # In the future convert it in test set
                for i in range(0, TRAIN_SIZE, RECORDING_STEP):
                    img = data[[i]]
                    groundtruth = labels[[i]]
                    accuracy = self.output_stats(sess, img, groundtruth)
                    print('Image' + str(i+1) + ' accuracy:', accuracy, '%%')
                    self.store_prediction(sess, img, groundtruth,\
                                            save_path = OUTPUT_PATH + "train_predictions/image_{a}_epoch_{b}.png".format(a=(i+1), b=epoch))

            return model_path


    def output_stats(self, sess, batch_x, batch_y):
        # Calculate batch loss and accuracy
        loss, acc, predictions = sess.run([self.conv_net.cost, self.conv_net.accuracy, self.conv_net.predicter],
                                                        feed_dict={self.conv_net.x: batch_x, self.conv_net.y: batch_y, self.conv_net.keep_prob: 1.,\
                                                                    self.conv_net.phase_train: False})

        return utils.accuracy(predictions, batch_y)



    def store_prediction(self, sess, test_x, test_y, save_path):
        predictions = sess.run(self.conv_net.predicter, feed_dict={self.conv_net.x: test_x, self.conv_net.y: test_y, self.conv_net.keep_prob: 1.,\
                                                                    self.conv_net.phase_train: False})

        utils_img.compare_proba_pred(predictions, np.squeeze(test_x, axis = 0), save_path)




def main():
    conv_net = ConvNet()
    #trainer = Trainer(conv_net)
    #save_model_path = trainer.train(restore = False)
    #print('Model saved in:', save_model_path)

    data = utils_img.load_images(TRAINING_PATH,1)
    predictions = conv_net.predict(data[[0]])
    utils_img.compare_proba_pred(predictions, data[0], OUTPUT_PATH + 'prediction_satImage_001.png')


if __name__ == '__main__':
    main()
