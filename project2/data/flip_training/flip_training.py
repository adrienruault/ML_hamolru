
import numpy as np
import matplotlib.image as mpimg
from scipy import misc
import os



TRAINING_PATH = '../data/training/images/'
GROUNDTRUTH_PATH = '../data/training/groundtruth/'

TRAIN_SIZE = 100


def flip_training():
    # load raw imgs
    data = load_images(TRAINING_PATH, TRAIN_SIZE)

    groundtruth = load_groundtruths(GROUNDTRUTH_PATH, TRAIN_SIZE)


    if not os.path.exists('./images/'):
        os.makedirs('./images/')

    if not os.path.exists('./groundtruth/'):
        os.makedirs('./groundtruth/')


    for i in range(TRAIN_SIZE):

        data_img = data[i]
        groundtruth_img = groundtruth[i].reshape(400, 400)
        print(groundtruth_img.shape)
        data_name = "./images/satImage_%.3d" % (i*4 + 1) + '.png'
        groundtruth_name = "./groundtruth/satImage_%.3d" % (i*4 + 1) + '.png'
        misc.imsave(data_name, data_img)
        misc.imsave(groundtruth_name, groundtruth_img)
        for j in range(3):
            data_img = np.rot90(data_img)
            groundtruth_img = np.rot90(groundtruth_img)
            data_name = "./images/satImage_%.3d" % (i*4 + 2 + j) + '.png'
            groundtruth_name = "./groundtruth/satImage_%.3d" % (i*4 + 2 + j) + '.png'
            misc.imsave(data_name, data_img)
            misc.imsave(groundtruth_name, groundtruth_img)







def load_images(folder_path, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
        Indices are from 0.
        Values are rescaled from [0, 255] down to [0.0, 1.0]. """
    imgs = np.zeros(shape=[num_images, 400, 400, 3])
    for i in range(1, num_images + 1):
        image_name = "satImage_%.3d" % i
        image_path = folder_path + image_name + ".png"
        if os.path.isfile(image_path):
            print('Loading ' + image_path)
            img = mpimg.imread(image_path)

            #imgs[i - 1] = np.asarray(img).reshape(400, 400, 3)
            imgs[i - 1] = img.reshape(400, 400, 3)
        else:
            print('File ' + image_path + ' does not exist')
    return imgs







def load_groundtruths(folder_path, num_images):
    """Extract the images into a 4D tensor [image index, y, x, channels].
        Indices are from 0.
        Values are rescaled from [0, 255] down to [0.0, 1.0]. """
    imgs = np.zeros(shape=[num_images, 400, 400, 1])
    for i in range(1, num_images + 1):
        image_name = "satImage_%.3d" % i
        image_path = folder_path + image_name + ".png"
        if os.path.isfile(image_path):
            print('Loading ' + image_path)
            img = mpimg.imread(image_path)

            #imgs[i - 1] = np.asarray(img).reshape(400, 400, 3)
            imgs[i - 1] = img.reshape(400, 400, 1)
        else:
            print('File ' + image_path + ' does not exist')
    return imgs







if __name__ == '__main__':
    flip_training()
