import numpy as np
from scipy import misc
import os
import matplotlib.image as mpimg







def convert_test_set_to_good_format():
    n_images = 50
    imgs = np.zeros(shape=[n_images, 608, 608, 3])
    imgs_converted = np.zeros(shape=[200, 400, 400, 3])
    for i in range(1, n_images +1):
        img_path = './data/test_set_images/test_' + str(i) + '/test_' + str(i) + '.png'
        if os.path.isfile(img_path):
            print('Loading ' + img_path)
            img = mpimg.imread(img_path)
            imgs[i-1] = img.reshape(608, 608, 3)
            imgs_converted[(i-1)*4] = (imgs[i-1, 0:400, 0:400, 0:3]).reshape(400, 400, 3)
            imgs_converted[(i-1)*4+1] = (imgs[i-1, 208:608, 0:400, 0:3]).reshape(400, 400, 3)
            imgs_converted[(i-1)*4+2] = (imgs[i-1, 0:400, 208:608, 0:3]).reshape(400, 400, 3)
            imgs_converted[(i-1)*4+3] = (imgs[i-1, 208:608, 208:608, 0:3]).reshape(400, 400, 3)

            misc.imsave('./data/test_set_good_format/test_' + str(i) + '_' + str(1) + '.png', imgs_converted[(i-1)*4])
            misc.imsave('./data/test_set_good_format/test_' + str(i) + '_' + str(2) + '.png', imgs_converted[(i-1)*4+1])
            misc.imsave('./data/test_set_good_format/test_' + str(i) + '_' + str(3) + '.png', imgs_converted[(i-1)*4+2])
            misc.imsave('./data/test_set_good_format/test_' + str(i) + '_' + str(4) + '.png', imgs_converted[(i-1)*4+3])








if __name__ == '__main__':
    convert_test_set_to_good_format()
