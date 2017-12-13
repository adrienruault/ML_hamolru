import os
import matplotlib.image as mpimg
from PIL import Image

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


def load_groundtruths(folder_path, num_images):
    """Extract the groundtruth images into a 4D tensor [image index, y, x, channels].
        Indices are from 0."""
    imgs = np.zeros(shape=[num_images, 400, 400, 1])
    for i in range(1, num_images + 1):
        image_name = "satImage_%.3d" % i
        image_path = folder_path + image_name + ".png"
        if os.path.isfile(image_path):
            print('Loading ' + image_path)
            img = mpimg.imread(image_path)
            print('type', type(img))
            imgs[i - 1] = img.reshape(400, 400, 1)
        else:
            print('File ' + image_path + ' does not exist')
    imgs = np.around(imgs)
    return imgs


def mask_to_img_values(img):
    """ Rescales pixel values between 0 and 255"""
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * 255).round().astype(np.uint8)
    return rimg


def main():
    imgs = load_groundtruths("./data/training/groundtruth/", 10)
    print(imgs.shape)
    print(imgs[0, 190:210, 90:100])
    Image.fromarray(mask_to_img_values(imgs[0].reshape([400, 400]))).convert("L").save("test.png")


if __name__ == '__main__':
    main()
