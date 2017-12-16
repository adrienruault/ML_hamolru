import os
import matplotlib.image as mpimg
from scipy import misc

import numpy as np


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



def save_image_from_proba_pred(prediction, save_path):
    img_prediction = np.argmax(prediction, axis = 3)
    img_prediction = np.squeeze(img_prediction, axis = 0)
    #print(img_prediction)
    img_prediction = img_prediction * 255
    #print('img_prediction shape:', img_prediction.shape)
    #print(img_prediction)
    misc.imsave(save_path, img_prediction)
