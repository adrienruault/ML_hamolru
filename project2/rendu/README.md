# Machine Learning Project: Road Segmentation

## Parameters

Our submission has been ran with the following parameters : 

* Training size: 400 images
* Batch size: 2 images
* Number of epochs: 8
* Regularization parameter: 1e-15
* Dropout: 0 %

## Structure

### data

#### data/training

Put the satellite images of the training set in the data/training/images folder and the ground-truth images in the data/training/groundtruth folder.

#### data/test_set_images

This is where the satellite images from the test set should be put.

### submitted_conv_net_model

Contains the precomputed results that are utilized to create the Kaggle submission in csv format.

### run.py

Run this file to create the submission : Use the command "python run.py" in the terminal. Set restore_flag = True to get the precomputed submission we used (default value). Set restore_flag = False to retrain on the train set.

You must have python 3.6 in order for the code to work correctly with the latest version of TensorFlow installed.

### conv_net.py	

Used by run.py to apply the model to the test set and get predictions

### test_set_formatting.py

Used by run.py to convert the test set to the same format as the training set (i.e. 400 x 400 x 3).

### flip_training.py

Used by run.py to augment the training set by flipping images.

### utilities.py & utilities_image.py

Contains utilitary functions such used in the convolutional network and for image processing.

### Group members:

Skander Hajri

Guillaume Mollard

Adrien Ruault

### Team name on Kaggle : "Saucisson Vin Rouge"