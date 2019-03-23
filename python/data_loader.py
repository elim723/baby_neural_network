#!/usr/bin/env python3

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

    ''' A class to load label and image data

        Examples
        --------
        In [1]: import data_loader
        In [2]: data = data_loader.data_loader ()
        In [3]: data.set_train_subset (100, random_sample=True)
        In [4]: training_images = data.train_images
        In [4]: data.display_images (data.train_images, plot_name='train.png')
    '''

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
        self._n_classes = 10         ## an image must belong to one of the 10 digits 0-9
        self._max_pixel_intensity = 255
        ### set up properties accessible by public
        self._train_sample_indices = None
        self._test_sample_indices = None

    def __str__ (self):

        try:
            ### used when subset of training / testing samples are sampled
            return '{0} ({1}) training (testing) examples are sampled from a total of {2} ({3}) training (testing) examples.'.format (self.n_train_samples, self.n_test_samples, self.max_n_train_samples, self.max_n_test_samples)
        except:
            try:
                ### used when subset of training samples are sampled, but not testing
                return '{0} training examples are sampled from a total of {1} training examples.'.format (self.n_train_samples, self.max_n_train_samples)
            except:
                try:
                    ### used when subset of testing samples are sampled, but not training
                    return '{0} testing examples are sampled from a total of {1} testing examples.'.format (self.n_test_samples, self.max_n_test_samples)
                except:
                    try:
                        ### used when both training and testing sets are loaded
                        return 'A total of {0} ({1}) training (testing) examples is loaded. No sampling was done'.format (self.max_n_train_samples, self.max_n_test_samples)
                    except:
                        try:
                            ### used when training but not testing is loaded
                            return 'A total of {0} training examples is loaded. No sampling was done'.format (self.max_n_train_samples)
                        except:
                            try:
                                ### used when testing but not training is loaded
                                return 'A total of {0} testing examples is loaded. No sampling was done'.format (self.max_n_test_samples)
                            except:
                                ### used when nothing is loaded
                                return 'No training / testing examples.'

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
        if self._full_train_labels is None:
            raise UnboundLocalError ('Full training samples are not loaded yet.')
        return len (self._full_train_labels)

    @property
    def max_n_test_samples (self):
        if self._full_test_labels is None:
            raise UnboundLocalError ('Full testing samples are not loaded yet.')
        return len (self._full_test_labels)

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
        if self._train_sample_indices is None:
            raise UnboundLocalError ('Training examples are not sampled yet.')
        return len (self._train_sample_indices)

    @property
    def n_test_samples (self):
        if self._test_sample_indices is None:
            raise UnboundLocalError ('Testing examples are not sampled yet.')
        return len (self._test_sample_indices)

    @property
    def train_images (self):
        if self._train_sample_indices is None:
            raise UnboundLocalError ('Training examples are not sampled yet.')
        return self._full_train_images[self._train_sample_indices]

    @property
    def train_labels (self):
        if self._train_sample_indices is None:
            raise UnboundLocalError ('Training examples are not sampled yet.')
        return self._full_train_labels[self._train_sample_indices]

    @property
    def test_images (self):
        if self._test_sample_indices is None:
            raise UnboundLocalError ('Testing examples are not sampled yet.')
        return self._full_test_images[self._test_sample_indices]

    @property
    def test_labels (self):
        if self._test_sample_indices is None:
            raise UnboundLocalError ('Testing examples are not sampled yet.')
        return self._full_test_labels[self._test_sample_indices]

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

        try:

            max_n_samples = self.max_n_test_samples if is_test else \
                            self.max_n_train_samples

            ## if value is less than 1, it is a percentage
            ## else it is the actual number of training samples
            n_samples = int (max_n_samples * value) if value <= 1.0 else value

            ## make sure the input n samples is an integer
            if not type (n_samples) == int:
                variable = 'n_test_samples' if is_test else 'n_train_samples'
                message = variable + ' must be either a float <= 1.0 or an integer.'
                raise TypeError (message)

            ## make sure the input n samples is less than max
            if n_samples > max_n_samples:
                variable = 'n_test_samples' if is_test else 'n_train_samples'
                message = variable + ' cannot be larger than ' + str (max_n_samples) + '.'
                raise ValueError (message)

            return n_samples

        except:
            ### input n_samples is not the right type
            variable = 'n_test_samples' if is_test else 'n_train_samples'
            message = variable + ' must be either a float <= 1.0 or an integer.'
            raise TypeError (message)

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

    def load_test_samples (self):

        ''' A publicly accessible function to load all testing samples. '''

        ### load all testing sample
        images, labels = self._load_data (self._image_test_filename, self._label_test_filename)

        ### store them in self
        self._full_test_images, self._full_test_labels = images, labels

    def _get_random_indices (self, n_samples, max_n_samples):

        ''' A private function to get random indices

            input params
            ------------
                n_samples (int): number of sampled examples
                max_n_samples (int): maximum number of examples to be sampled from

            output params
            -------------
                sampled_indices (np.array): selected indices
        '''

        population = range (max_n_samples)

        ### randomly select the training sam = ples by the indices
        try:
            ## if python 3+
            return random.choices (population, k=n_samples)
        except:
            ##  if python 2.7
            return random.sample (population, k=n_samples)

    def _get_sampled_indices (self, n_samples, is_test=False, random_sample=False):

        ''' A private function returning the sampled indices

            input params
            ------------
                n_samples (int): number of examples to be sampled
                is_test  (bool): If True, randomly sample from full test examples.
                                 By default, randomly sample from full train examples.
                random_sample (bool): If True, randomly sample from full samples

            output params
            -------------
                sampled_indices (array): sampled indices from full sample
        '''

        ### define variables based on training / testing samples
        max_n_samples = self.max_n_test_samples if is_test else self.max_n_train_samples

        sampled_indices = self._get_random_indices (n_samples, max_n_samples) \
                          if random_sample else range (n_samples)

        ### return the sampled indices / images / labels
        return sampled_indices

    def set_train_subset (self, n_samples, random_sample=False):

        ''' A public function set a subset of training sample.

            input params
            ------------
                n_samples (float / int): If float less than 1.0, its percentage.
                                         If int, it is actual sample size.
                                         This will reset self._n_train_samples
                random_sample (bool): If True, randomly sample from full samples
        '''

        ### load training samples if not already
        if self._full_train_images is None or self._full_train_labels is None:
            self.load_train_samples ()
        ### make sure input n_samples is legit
        n_samples = self._check_n_samples (n_samples, is_test=False)

        ### get sampled indices
        self._train_sample_indices = \
            self._get_sampled_indices (n_samples, is_test=False,
                                       random_sample=random_sample)

    def set_test_subset (self, n_samples, random_sample=False):

        ''' A public function set a subset of testing sample.

            input params
            ------------
                n_samples (float / int): If float less than 1.0, its percentage.
                                         If int, it is actual sample size.
                                         This will reset self._n_train_samples
                random_sample (bool): If True, randomly sample from full samples
        '''

        ### load testing samples if not already
        if self._full_test_images is None or self._full_test_labels is None:
            self.load_test_samples ()
        ### make sure input n_samples is legit
        n_samples = self._check_n_samples (n_samples, is_test=True)

        ### get sampled indices
        self._test_sample_indices = \
            self._get_sampled_indices (n_samples, is_test=True,
                                       random_sample=random_sample)

    def display_images (self, images,
                        plot_name = default_plot_folder + 'images.pdf'):

        ''' A public function to display input images

            input params
            ------------
                images  (array): a n_images x n_row x n_col array with
                                 intensities of all pixels
                plot_name (str): path / name of output plot
        '''

        ### figure size is 7.5 inches width x 5.5 inches height
        h = plt.figure (figsize=(7.5, 5.5))

        ### divide plot into sub plots
        n_samples_per_axis = int (np.ceil (np.sqrt (len (images))))
        gs = gridspec.GridSpec (n_samples_per_axis, n_samples_per_axis)
        gs.update (wspace=0.1, hspace=0.1)

        ### set x / y edges
        xedges = np.arange (0, self.n_pixels_per_col+1)
        xedges = xedges[:-1] + 0.5*(xedges[1:] - xedges[:-1])
        yedges = np.arange (0, self.n_pixels_per_row+1)
        yedges = yedges[:-1] + 0.5*(yedges[1:] - yedges[:-1])

        ##  loop through each selected image and plot
        for index, image in enumerate (images):

            # location of the index-th subplot
            axis = h.add_subplot (gs[index])

            # plot intensity; inverted so it is not up side down
            axis.pcolormesh (xedges, yedges, image[::-1],
                             vmin=0, vmax=1, cmap=plt.get_cmap ('gray'))

            # hide all axes ticks
            axis.get_xaxis ().set_ticks ([])
            axis.get_yaxis ().set_ticks ([])

        ### save plot
        plt.suptitle (str (len (images)) + ' examples of hand written digits')
        h.savefig (plot_name)
        plt.close ('all')
