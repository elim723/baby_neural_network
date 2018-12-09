####
#### By Elim Thompson (12/06/2018)
####
#### This data loader class loads the hand written images, as well
#### as the corresponding labels. This class has the options to
####   1. load training or testing sets
####   2. return only a certain number or percentage of data
####   3. randomly sample from all data
####   4. display the selected sample
####
#### Here are some background info about the data itself:
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
############################################################################

### import packages
import numpy as np
import struct, random, warnings, matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rc ('text', usetex=True)
plt.rc ('font', family='sans-serif')
plt.rc ('font', serif='Computer Modern Roman')

### +------------------------------------------
### | Define default locations
### +------------------------------------------
default_work_folder = '/home/elims/elimstuff/programming/projects/nn_digits/'
default_plot_folder = default_work_folder + 'scripts/plots/'
default_label_train_filename = default_work_folder + 'datasets/train-labels-idx1-ubyte'
default_image_train_filename = default_work_folder + 'datasets/train-images-idx3-ubyte'
default_label_test_filename  = default_work_folder + 'datasets/t10k-labels-idx1-ubyte'
default_image_test_filename  = default_work_folder + 'datasets/t10k-images-idx3-ubyte'

class data_loader (object):

    ''' A class to load label and image data '''

    def __init__ (self,
                  label_train_filename=default_label_train_filename,
                  image_train_filename=default_image_train_filename,
                  label_test_filename=default_label_test_filename,
                  image_test_filename=default_image_test_filename):

        ''' initialize a data loader class with filenames

            inputs params
            -------------
                label_train_filename (str): location of training labels
                image_train_filename (str): location of training images
                label_test_filename  (str): location of testing labels
                image_test_filename  (str): location of testing images
        '''

        ### set up locations
        self._label_train_filename = label_train_filename
        self._image_train_filename = image_train_filename
        self._label_test_filename = label_test_filename
        self._image_test_filename = image_test_filename
        ### other internal private properties
        self._full_train_images, self._full_test_images = None, None
        self._full_train_labels, self._full_test_labels = None, None
        self._n_pixels_per_row   = 0
        self._n_pixels_per_col   = 0
        self._max_n_train_samples = 0
        self._max_n_test_samples  = 0
        self._n_classes = 10         ## an image must belong to one of the 10 digits 0-9
        self._max_pixel_intensity = 255
        ### set up properties accessible by public
        self._n_train_samples = 1
        self._n_test_samples = 1

    @property
    def full_train_images (self):
        return self._full_train_images

    @property
    def full_train_labels (self):
        return self._full_train_labels

    @property
    def full_test_images (self):
        return self._full_test_images

    @property
    def full_test_labels (self):
        return self._full_test_labels

    @property
    def n_classes (self):
        return self._n_classes

    @property
    def max_n_train_samples (self):
        return self._max_n_train_samples

    @property
    def max_n_test_samples (self):
        return self._max_n_test_samples

    @property
    def n_pixels_per_row (self):
        return self._n_pixels_per_row

    @property
    def n_pixels_per_col (self):
        return self._n_pixels_per_col

    @property
    def n_pixels (self):
        return self._n_pixels_per_row * self._n_pixels_per_col

    @property
    def n_train_samples (self):
        return self._n_train_samples

    @n_train_samples.setter
    def n_train_samples (self, value):
        try:
            self._n_train_samples = self._check_n_samples (value, is_test=False)
        except:
            message = 'n_train_samples must be either a float <= 1.0 or an integer.'
            raise TypeError (message)

    @property
    def n_test_samples (self):
        return self._n_test_samples

    @n_test_samples.setter
    def n_test_samples (self, value):
        try:
            self._n_test_samples = self._check_n_samples (value, is_test=True)
        except:
            message = 'n_test_samples must be either a float <= 1.0 or an integer.'
            raise TypeError (message)

    def _check_n_samples (self, value, is_test=False):

        ''' A private function to check the input values for either
            n_train_samples or n_test_samples. If value is less than
            1.0, it is interpreted as a percentage; otherwise, it
            is the actual number of samples.

            input params
            ------------
                value   (int or float): user input value
                is_test (bool): If True, checking test samples.
                                check train samples by default.

            output params
            -------------
                n_samples (int): number of test/train samples to be included
        '''

        ## if value is less than 1, it is a percentage
        ## else it is the actual number of training samples
        n_samples = int (max_n_samples * value) if value <= 1.0 else value

        ## make sure the input n samples is an integer
        if not type (n_samples) == int:
            variable = 'n_test_samples' if is_test else 'n_train_samples'
            message = variable + ' must be either a float <= 1.0 or an integer.'
            raise TypeError (message)

        ## make sure the input n samples is less than max
        max_n_samples = self._max_n_train_samples if is_test else \
                        self._max_n_train_samples
        if n_samples > max_n_samples:
            variable = 'n_test_samples' if is_test else 'n_train_samples'
            message = variable + ' cannot be larger than ' + str (max_n_samples) + '.'
            raise ValueError (message)

        return n_samples

    def _load_data (self, image_filename, label_filename):

        ''' A private function to load data based on current values of
            random_sampling and n_samples.

            input params
            ------------
                label_filename (str): location of label file
                image_filename (str): location of image file

            output params
            -------------
                images (array): normalized intensities in n_images x
                                n_pixels_per_rows x n_pixels_per_cols
                labels (array): true digits of loaded images
        '''

        ### load label
        with open (label_filename, 'rb') as f:
            magic_label, n_labels = struct.unpack ('>II',f.read (8))
            labels = np.fromfile (f, dtype=np.uint8)
        f.close ()

        ### load image
        with open (image_filename, 'rb') as f:
            magic_image, n_images, n_rows, n_cols = struct.unpack ('>IIII',f.read (16))
            images = np.fromfile (f, dtype=np.uint8)
        f.close ()

        ### warn users that n_pixels_per_row / col are reset if different from before
        if not self._n_pixels_per_row == n_rows:
            warnings.warn ('number of pixels per row per image is changed from ' +
                           str (self._n_pixels_per_row) + ' to ' + str (n_rows) + '.')
        if not self._n_pixels_per_col == n_cols:
            warnings.warn ('number of pixels per column per image is changed from ' +
                           str (self._n_pixels_per_col) + ' to ' + str (n_cols) + '.')
        ##  reset number of pixels per row / col per image
        self._n_pixels_per_row = n_rows
        self._n_pixels_per_col = n_cols

        ### reshape images and normalize values to max value of 255
        images = images.reshape (n_images, n_rows, n_cols) / self._max_pixel_intensity

        return images, labels

    def load_train_samples (self):

        ''' A publicly accessible function to load all training samples. '''

        ### load all training sample
        images, labels = self._load_data (self._image_train_filename, self._label_train_filename)

        ### store them in self
        self._full_train_images, self._full_train_labels = images, labels
        self._max_n_train_samples = len (labels)

    def load_test_samples (self):

        ''' A publicly accessible function to load all testing samples. '''

        ### load all testing sample
        images, labels = self._load_data (self._image_test_filename, self._label_test_filename)

        ### store them in self
        self._full_test_images, self._full_test_labels = images, labels
        self._max_n_test_samples = len (labels)

    def _get_random_indices (self, n_samples, is_test=False):

        max_n_samples = self._max_n_test_samples if is_test else \
                        self._max_n_train_samples

        population = range (max_n_samples)

        ### randomly select the training sam = ples by the indices
        try:
            ## if python 3+
            sampled_indices = random.choices (population, k=n_samples)
        except:
            ##  if python 2.7
            sampled_indices = random.sample (population, k=n_samples)
        return sampled_indices

    def _get_sample (self, is_test=False, random_sample=False):

        ''' A private function returning a subset of training / testing sample.

            input params
            ------------
                is_test (bool): If True, randomly sample from full test examples.
                                By default, randomly sample from full train examples.
                random_sample (bool): If True, randomly sample from full samples

            output params
            -------------
                sampled_images (array): sampled images with a size of n_samples x
                                        n_pixels_per_rows x n_pixels_per_cols
                sampled_labels (array): sampled labels of true digits of loaded images
        '''

        ### define variables based on training / testing samples
        if is_test:
            full_images, full_labels = self._full_test_images, self._full_test_labels
            n_samples = self.n_test_samples
        else:
            full_images, full_labels = self._full_train_images, self._full_train_labels
            n_samples = self.n_train_samples

        sampled_indices = self._get_random_indices (n_samples, is_test=is_test) \
                          if random_sample else range (n_samples)

        ### actually get the images and labels
        sampled_images = full_images [sampled_indices]
        sampled_labels = full_labels [sampled_indices]

        ### return the sampled images / labels
        return sampled_images, sampled_labels

    def get_train_samples (self, n_samples, random_sample=False):

        ''' A public function returning a subset of training sample.

            input params
            ------------
                n_samples (float / int): If float less than 1.0, its percentage.
                                         If int, it is actual sample size.
                                         This will reset self._n_train_samples
                random_sample (bool): If True, randomly sample from full samples

            output params
            -------------
                sampled_images (array): sampled training images with a size of
                                        n_samples x n_pixels_per_rows x n_pixels_per_cols
                sampled_labels (array): sampled labels of true digits of loaded
                                        training images
        '''

        ### load training samples if not already
        if self._full_train_images is None or self._full_train_labels is None:
            self.load_train_samples ()
        ### set n_training_samples
        self.n_train_samples = n_samples

        return self._get_sample (is_test=False, random_sample=random_sample)

    def get_test_samples (self, n_samples, random_sample=False):

        ''' A public function returning a subset of testing sample.

            input params
            ------------
                n_samples (float / int): If float less than 1.0, its percentage.
                                         If int, it is actual sample size.
                                         This will reset self._n_test_samples
                random_sample (bool): If True, randomly sample from full samples

            output params
            -------------
                sampled_images (array): sampled testing images with a size of
                                        n_samples x n_pixels_per_rows x n_pixels_per_cols
                sampled_labels (array): sampled labels of true digits of loaded
                                        testing images
        '''

        ### load testing samples if not already
        if self._full_test_images is None or self._full_test_labels is None:
            self.load_test_samples ()
        ### set n_testing_samples if not already
        self.n_test_samples = n_samples

        return self._get_sample (is_test=True, random_sample=random_sample)
