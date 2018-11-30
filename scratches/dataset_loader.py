####
#### By Elim Thompson (11/29/2018)
####
#### This scratch script is my play area to just get a taste of the
#### data sets for my Devember 2018 project - writing a neutral
#### network from scratch to recognize hand written digits using
#### python with numpy / scipy / matplotlib.
####
#### The public hand written digit datasets are downloaded from
####    http://yann.lecun.com/exdb/mnist/
####
#### The training set has 60k examples, whereas the testing set has
#### 10k. Training set will be used for the algorithm development,
#### and testing will not be used until the very end for performance
#### analysis.
####
#### The training set has two zip files: a label file and an image
#### files. Both files are binary and can be unpacked via the struct
#### package in python. Here are the information copied from the
#### website above:
####
#### For the label file
#### [offset] [type]          [value]          [description]
#### 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
#### 0004     32 bit integer  60000            number of items
#### 0008     unsigned byte   ??               label
####
#### For the image file
#### [offset] [type]          [value]          [description]
#### 0000     32 bit integer  0x00000803(2051) magic number
#### 0004     32 bit integer  60000            number of images
#### 0008     32 bit integer  28               number of rows
#### 0012     32 bit integer  28               number of columns
#### 0016     unsigned byte   ??               pixel
####
#### These info will be useful when using struct.unpack to load the
#### labels and images from the files.
####
#### After unpacked, `images` is the `flattened` variable that contains
#### all pixels from all training images. There are 60k images, each of
#### which are 28 x 28 pixels, and the intensity of each pixel ranges
#### from 0 (white) to 255 (black). Similarly, `labels` is a 60k-element
#### array with the true answers of the training images. That is, each
#### element of the label array, ranging from 0 to 9, is the answer of
#### the corresponding image in the images array.
####
########################################################################

### import packages
import numpy as np
import struct

### define dataset locations
dataset_folder = '/home/elims/elimstuff/programming/projects/nn_digits/datasets/'
label_filename = dataset_folder + 'train-labels-idx1-ubyte'
image_filename = dataset_folder + 'train-images-idx3-ubyte'

### deal with label file
with open (label_filename, 'rb') as f:
    ## According to the source website, the first two pieces of data
    ## are magic number and number of items; need 8 bytes
    magic_label, n_labels = struct.unpack ('>II',f.read(8))
    ## Reading each label as uint8 is plenty sufficient
    ## because labels range from 0 to 9
    labels = np.fromfile (f, dtype=np.uint8)
f.close ()
##  print the basic info from label file
print (magic_label, n_labels) # (2049, 60000)
print (len (labels))          # 60000
print (np.unique (labels))    # [0 1 2 3 4 5 6 7 8 9]

### deal with image file
with open (image_filename, 'rb') as f:
    ## According to the source website, the first four pieces of data
    ## are magic number, number of items, number of rows and columns; need 16 bytes
    magic_image, n_images, n_rows, n_cols = struct.unpack ('>IIII',f.read(16))
    ## Reading each image as uint8 is sufficient
    ## because each pixel ranges from 0 to 255
    images = np.fromfile (f, dtype=np.uint8)
f.close ()
##  print the basic info from image file
print (magic_image, n_images, n_rows, n_cols) # (2051, 60000, 28, 28)
print (len (images))                          # 47040000
print (np.min (images), np.max (images))      # (0, 255)

### reshape `images` to a n_images x n_rows x n_cols matrix
images = images.reshape (n_images, n_rows, n_cols)
print (images.shape) # (60000, 28, 28)
