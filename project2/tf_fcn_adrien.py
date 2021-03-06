
import preprocessing
import tensorflow as tf
#from tensorflow.examples.tutorials.mnist import input_data
import random
#mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# To be changed to 2
NUM_CLASSES = 2

# To be changed to 400, 400
IMG_WIDTH = 400
IMG_HEIGHT = 400

NUM_EPOCHS = 2

# To be changed to 3
NUM_CHANNELS = 3

# To be changed to 1
BATCH_SIZE = 128

TRAIN_SIZE = 20
TRAIN_PATH = './training/training/'
LABEL_PATH = './training/images/'

SEED = 1

random.seed(SEED)


# These values need to be fed in the sess.run() function using feed_dict
x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, 3], name="input_image")
y = tf.placeholder(tf.int32, shape=[BATCH_SIZE, IMG_WIDTH, IMG_HEIGHT, 1], name="annotation")


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding = 'SAME')


# ksize defines the size of the pool window
# strides defines the way the window moves (movement of the window)
def maxpool2d(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')


KEEP_RATE = 0.8
KEEP_PROB = tf.placeholder(tf.float32)


# data must be a 4D tensor generated by tf.constant()
# applied on a 4D numpy array with [image index, IMAGE_WIDTH, IMAGE_HEIGHT, CHANNEL]
def FCN_model(data):

    print('data shape:', data.shape)
    # Could use tf.truncated_normal() instead of tf.random_normal(),
    # see what is the best choice
    # note tf.truncated() is used in the FCN implementation
    weights = {'W_conv1_1': tf.Variable(tf.truncated_normal([3, 3, NUM_CHANNELS, 64])),
               'W_conv1_2': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
               'W_conv2_1': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
               'W_conv2_2': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
               'W_conv3_1': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
               'W_conv3_2': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
               'W_conv4_1': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
               'W_conv4_2': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
               'W_conv5_1': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
               'W_conv5_2': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
               'W_conv6': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
               'W_conv7': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
               # Number of pools done until now: 2 -> image size is (IMG_WIDTH / 4 * IMG_HEIGHT / 4)
               'out': tf.Variable(tf.truncated_normal([1024, NUM_CLASSES]))}

    biases = {'B_conv1_1': tf.Variable(tf.truncated_normal([3, 3, NUM_CHANNELS, 64])),
              'B_conv1_2': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
              'B_conv2_1': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
              'B_conv2_2': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
              'B_conv3_1': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
              'B_conv3_2': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
              'B_conv4_1': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
              'B_conv4_2': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
              'B_conv5_1': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
              'B_conv5_2': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
              'B_conv6': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),
              'B_conv7': tf.Variable(tf.truncated_normal([3, 3, 64, 64])),}

    # IMPORTANT STEP
    # From FCN implementation:
    # shape = tf.shape(data)
    # deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
    data = tf.reshape(data, shape=[-1, IMG_WIDTH, IMG_HEIGHT, NUM_CHANNELS])

    conv1 = conv2d(data, weights['W_conv1'])
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, biases['B_conv1']))
    pool1 = maxpool2d(relu1)

    conv2 = conv2d(pool1, weights['W_conv2'])
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, biases['B_conv2']))
    pool2 = maxpool2d(conv2)

    dropout = tf.nn.dropout(pool2, KEEP_RATE)

    output = tf.matmul(dropout, weights['out'] + biases['out'])

    return output





def train_neural_network(x):
    # the prediction variable is often called logits in tensor flow vocabulary
    prediction = FCN_model(x)

    print('prediction shape', prediction.shape)
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y) )
    cost = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction,
                                                                        labels=tf.squeeze(y, squeeze_dims=[3]),
                                                                        name="entropy")))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Loading train with labels
    img_train_list = preprocessing.load_images(TRAIN_PATH, TRAIN_SIZE)
    label_train_list = preprocessing.load_groundtruths(LABEL_PATH, TRAIN_SIZE)

    batch_idx = range(int(TRAIN_SIZE / BATCH_SIZE))

    hm_epochs = NUM_EPOCHS
    with tf.Session() as sess:
        # initialization of the variables (eg: truncated normal)
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_batch_idx = random.shuffle(batch_idx)
            epoch_loss = 0
            for batch in range(int(NUM_IMAGES_TRAIN/BATCH_SIZE)):
                # feeding epoch_x and epoch_y for training current batch (to replace with our own )
                epoch_x = img_train_list[batch_idx[batch]]
                epoch_y = label_train_list[batch_idx[batch]]
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
