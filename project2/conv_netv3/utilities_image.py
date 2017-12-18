import os
import matplotlib.image as mpimg
from scipy import misc
import tensorflow as tf

import numpy as np



PIXEL_DEPTH = 255


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


def convert_test_set_to_good_format():
    n_images = 50
    imgs = np.zeros(shape=[n_images, 608, 608, 3])
    imgs_converted = np.zeros(shape=[200, 400, 400, 3])
    for i in range(1, n_images +1):
        img_path = '../data/test_set_images/test_' + str(i) + '/test_' + str(i) + '.png'
        if os.path.isfile(img_path):
            print('Loading ' + img_path)
            img = mpimg.imread(img_path)
            imgs[i-1] = img.reshape(608, 608, 3)
            imgs_converted[(i-1)*4] = (imgs[i-1, 0:400, 0:400, 0:3]).reshape(400, 400, 3)
            imgs_converted[(i-1)*4+1] = (imgs[i-1, 208:608, 0:400, 0:3]).reshape(400, 400, 3)
            imgs_converted[(i-1)*4+2] = (imgs[i-1, 0:400, 208:608, 0:3]).reshape(400, 400, 3)
            imgs_converted[(i-1)*4+3] = (imgs[i-1, 208:608, 208:608, 0:3]).reshape(400, 400, 3)

            misc.imsave('../data/test_set_good_format/test_' + str(i) + '_' + str(1) + '.png', imgs_converted[(i-1)*4])
            misc.imsave('../data/test_set_good_format/test_' + str(i) + '_' + str(2) + '.png', imgs_converted[(i-1)*4+1])
            misc.imsave('../data/test_set_good_format/test_' + str(i) + '_' + str(3) + '.png', imgs_converted[(i-1)*4+2])
            misc.imsave('../data/test_set_good_format/test_' + str(i) + '_' + str(4) + '.png', imgs_converted[(i-1)*4+3])



def convert_image_to_hot(img):
    hot_img = np.zeros([400, 400, 2], dtype = float)
    for i in range(hot_img.shape[0]):
        for j in range(hot_img.shape[1]):
            if img[i,j] < 0.5:
                hot_img[i,j,0] = 1.0
                hot_img[i,j,1] = 0.0
            else:
                hot_img[i,j,0] = 0.0
                hot_img[i,j,1] = 1.0

    return hot_img

def load_groundtruths(folder_path, num_images):
    """Extract the groundtruth images into a 4D tensor [image index, y, x, channels].
        Indices are from 0."""
    imgs = []
    for i in range(1, num_images + 1):
        image_name = "satImage_%.3d" % i
        image_path = folder_path + image_name + ".png"
        if os.path.isfile(image_path):
            print('Loading ' + image_path)
            img = mpimg.imread(image_path)
            # See if it is better to use dtype = int
            hot_img = convert_image_to_hot(img)
            imgs.append(hot_img)
        else:
            print('File ' + image_path + ' does not exist')
    #imgs = np.around(imgs) # Uncomment if we want to round values.
    imgs_array = np.asarray(imgs)
    return imgs_array



def compare_proba_pred(prediction, original_img, save_path):
    #img_prediction = np.argmax(prediction, axis = 3)
    img_prediction = prediction[:,:,:,1]
    img_prediction = np.squeeze(img_prediction, axis = 0)
    #print(img_prediction)
    print('img_prediction shape', img_prediction.shape)
    print(img_prediction)
    #print('img_prediction shape:', img_prediction.shape)
    #print(img_prediction)
    min_value = np.amin(img_prediction)
    max_value = np.amax(img_prediction)
    img_prediction = (img_prediction - min_value) * (255.0 / (max_value - min_value))

    concatenated = concatenate_images(original_img, img_prediction)

    misc.imsave(save_path, concatenated)





def concatenate_images(img, gt_img):
    nChannels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if nChannels == 3:
        cimg = np.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = np.zeros((w, h, 3), dtype=np.uint8)
        gt_img8 = img_float_to_uint8(gt_img)
        gt_img_3c[:,:,0] = gt_img8
        gt_img_3c[:,:,1] = gt_img8
        gt_img_3c[:,:,2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = np.concatenate((img8, gt_img_3c), axis=1)
    return cimg


def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg


# Make an image summary for 4d tensor image with index idx
def get_image_summary(img, idx = 0):
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    img_w = img.get_shape().as_list()[1]
    img_h = img.get_shape().as_list()[2]
    min_value = tf.reduce_min(V)
    V = V - min_value
    max_value = tf.reduce_max(V)
    V = V / (max_value*PIXEL_DEPTH)
    V = tf.reshape(V, (img_w, img_h, 1))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, (-1, img_w, img_h, 1))
    return V

if __name__ == '__main__':
    convert_test_set_to_good_format()
