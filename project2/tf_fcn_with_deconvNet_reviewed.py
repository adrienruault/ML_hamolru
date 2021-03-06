
import numpy as np
import tensorflow as tf
import preprocessingv2 as pre
from random import shuffle
from scipy import misc

# To be changed to 2
NUM_CLASSES = 2

# To be changed to 400, 400
IMG_WIDTH = 400
IMG_HEIGHT = 400

NUM_EPOCHS = 1

# To be changed to 3
NUM_CHANNELS = 3

# To be changed to 1
BATCH_SIZE = 1

TRAIN_SIZE = 20
TEST_SIZE = 10

LEARNING_RATE = 1e-4

#test_data  = pre.load_images("./data/test_set_images/")




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
    h = ((in_shape[1] - 1) * stride) + 2
    w = ((in_shape[2] - 1) * stride) + 2
    new_shape = [in_shape[0], h, w, W.shape[3]]
    output_shape = tf.stack(new_shape)
    deconv = tf.nn.conv2d_transpose(x, W, output_shape,
                                    strides=strides, padding='SAME', name = name)
    return tf.nn.relu(tf.nn.bias_add(deconv, B))

def FCN_model(x):
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
                   'W_conv9': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name = 'W_conv9'),
                   'W_conv10': tf.Variable(tf.truncated_normal([3, 3, 64, 64]), name = 'W_conv10'),
                   'W_convout': tf.Variable(tf.truncated_normal([1, 1, 64, NUM_CLASSES]), name = 'W_convout')}

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
                   'B_conv9': tf.Variable(tf.truncated_normal([64]), name = 'B_conv9'),
                   'B_conv10': tf.Variable(tf.truncated_normal([64]), name = 'B_conv10'),
                   'B_convout': tf.Variable(tf.truncated_normal([NUM_CLASSES]), name = 'B_convout')}

    # IMPORTANT STEP
    # From FCN implementation:
    # shape = tf.shape(data)
    # deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
    #data = tf.reshape(data, shape=[BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS])

    # Going down
    conv_relu1 = conv2d_relu(x, weights['W_conv1'], biases['B_conv1'])

    conv_relu2 = conv2d_relu(conv_relu1, weights['W_conv2'], biases['B_conv2'])

    max_pool1 = maxpool2d(conv_relu2)

    conv_relu3 = conv2d_relu(max_pool1, weights['W_conv3'], biases['B_conv3'])

    conv_relu4 = conv2d_relu(conv_relu3, weights['W_conv4'], biases['B_conv4'])

    max_pool2 = maxpool2d(conv_relu4)

    conv_relu5 = conv2d_relu(max_pool2, weights['W_conv5'], biases['B_conv5'])

    conv_relu6 = conv2d_relu(conv_relu5, weights['W_conv6'], biases['B_conv6'])

    # Going up
    deconv_relu1 = deconv2d_relu(conv_relu5, weights['W_deconv1'], biases['B_deconv1'], upscale_factor=2)

    conv_relu7 = conv2d_relu(deconv_relu1, weights['W_conv7'], biases['B_conv7'])

    conv_relu8 = conv2d_relu(conv_relu7, weights['W_conv8'], biases['B_conv8'])

    deconv_relu2 = deconv2d_relu(conv_relu4, weights['W_deconv2'], biases['B_deconv2'], upscale_factor=2)

    conv_relu9 = conv2d_relu(deconv_relu2, weights['W_conv9'], biases['B_conv9'])

    conv_relu10 = conv2d_relu(conv_relu9, weights['W_conv10'], biases['B_conv10'])

    convout = conv2d_relu(conv_relu10, weights['W_convout'], biases['B_convout'])

    return convout


# def shuffle_training_set(data, labels):
#
#
#
# def next_batch(data, labels, batch_size):



def main(argv=None):

    # Placeholders definition
    # These values need to be fed in the sess.run() function using feed_dict
    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS])
    y = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, NUM_CLASSES])
    flat_y = tf.placeholder(tf.float32, shape=[BATCH_SIZE * IMG_WIDTH * IMG_HEIGHT, NUM_CLASSES])
    # keep_prob is used to define the dropout rate
    # Set to 0.85 (or other) while training to
    keep_prob = tf.placeholder(tf.float32)



    # In the very beginning we define the OPERATIONS
    # logits (from model) / cost / optimizer / train_prediction / saver

    # the prediction variable is often called logits in tensor flow vocabulary
    # Do everything so that softmax is never applied to logits beforehands
    logits_op = FCN_model(x)

    # SESSION RUN OPERATION
    # reduce mean apparently compute mean over a dimension
    # WARNING: This op expects unscaled logits, since it performs a softmax on logits internally for efficiency.
    # Do not call this op with the output of softmax, as it will produce incorrect results.
    flat_logits = tf.reshape(tensor=logits_op, shape=(-1, NUM_CLASSES))
    cost_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, labels=flat_y))
    # Possible to add regularizers to cost_op in to avoid overfitting

    # SESSION RUN OPERATION
    # Is AdamOptimizer the best choice, they use MomentumOptimizer in the template
    # Apparently: Adam optimizer requires less parameter tuning to get good results
    optimizer_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost_op)
    #trainable_var = tf.trainable_variables()
    #optimizer_op = tf.train.AdamOptimizer(LEARNING_RATE)
    #grads_and_vars = optimizer_op.compute_gradients(cost_op, trainable_var)
    #optimizer_op.apply_gradients(grads_and_vars)


    # SESSION RUN OPERATION
    # Predictions for the minibatch, validation set and test set.
    predictions_op = tf.nn.softmax(logits_op)

    # Add ops to save and restore all the variables.
    saver_op = tf.train.Saver()

    print('-> Loading data:')
    data = pre.load_images("data/training/images/", TRAIN_SIZE)
    labels = pre.load_groundtruths("data/training/groundtruth/", TRAIN_SIZE)

    print("train data shape: ", data.shape)
    print("labels shape: ", labels.shape)

    hm_epochs = NUM_EPOCHS
    image_indices = np.array(range(data.shape[0]))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #print('global variables:', tf.GraphKeys.GLOBAL_VARIABLES)

        for epoch in range(hm_epochs):
            shuffle(image_indices)
            count_batch = 0
            epoch_loss = 0
            for batch_id in range(int(TRAIN_SIZE/BATCH_SIZE)):
                # feeding epoch_x and epoch_y for training current batch (to replace with our own )
                epoch_x = data[[image_indices[count_batch:count_batch+BATCH_SIZE]]]
                epoch_y = labels[[image_indices[count_batch:count_batch + BATCH_SIZE]]]
                count_batch += BATCH_SIZE
                epoch_y = np.reshape(epoch_y, [-1, NUM_CLASSES])

                # sees.run() evaluate a tensor
                # the first argument of sess.run() is an array corresponding to every operation needed
                # Feeding sess.run() with y is necessary because cost_op needs it
                _, c, train_predictions = sess.run([optimizer_op, cost_op, predictions_op], feed_dict={x: epoch_x, flat_y: epoch_y})
                epoch_loss += c
                print('Batch', batch_id+1, 'completed out of', int(TRAIN_SIZE/BATCH_SIZE), ', batch loss:', c)

            print('epoch_x shape:', epoch_x.shape)
            print('epoch_y shape:', epoch_y.shape)
            print('-> Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        # Save the variables to disk.
        save_path = saver_op.save(sess, "./saved_models/FCN_model.ckpt")
        print("Model saved in file: %s" % save_path)

        # Get predictions for training set
        correct = tf.equal(tf.argmax(predictions_op, axis=3), tf.argmax(y, axis=3))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        accuracy_images = np.zeros(TRAIN_SIZE)
        for i in range(0, TRAIN_SIZE):
            img = data[[i]]
            data_node = tf.constant(img, dtype = tf.float32)
            print('data_node shape:', data_node.shape)
            #model_output = tf.nn.softmax(FCN_model(data_node))
            #output_prediction = sess.run(predictions_op, feed_dict={x: img})
            output_prediction = predictions_op.eval({x : img})
            #output_prediction = sess.run(tf.nn.softmax(FCN_model(data_node)))
            img_prediction = tf.argmax(output_prediction, axis = 3).eval()

            img_prediction = np.squeeze(img_prediction, axis = 0)
            print(img_prediction)
            img_prediction = img_prediction * 255
            if (i == 0):
                print('img_prediction shape:', img_prediction.shape)
                print(img_prediction)
            img_name = 'prediction_' + str(i+1) + '.png'
            misc.imsave('./predictions_training/' + img_name, img_prediction)

            image_accuracy = accuracy.eval({x : img, y : labels[[i]]})
            print('Accuracy train_image_{}:'.format(i+1), image_accuracy)
        print('Average train accuracy per image:', accuracy_images.mean())




if __name__ == '__main__':
    tf.app.run()
