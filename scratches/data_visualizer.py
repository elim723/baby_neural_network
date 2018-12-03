####
#### By Elim Thompson (11/30/2018)
####
#### This is another scratch script to visualize the training dataset.
#### Yesterday, I have figured out how to load and read the training set
#### provided by
####       http://yann.lecun.com/exdb/mnist/
#### Today, I want to plot the hand written digits via matplotlib.
####
#### Note for today:
#### 1. out-of-the-box image is upside down. It doesn't matter for
####    training. Just need to keep in mind when displaying
#### 2. to randomly select a sub set (>1 element), python 3 has a 'new'
####    `random.choices` function instead of the 'old' `random.sample`
####    function in python 2.7.
#### 3. looping 100 image samples takes about 11 seconds. Might have
####    a way to do it without looping ?
####
################################################################################

### import packages
import numpy as np
import struct, random,  matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
plt.rc ('text', usetex=True)
plt.rc ('font', family='sans-serif')
plt.rc ('font', serif='Computer Modern Roman')

### define dataset locations
plot_folder    = '/home/elims/elimstuff/programming/projects/nn_digits/scripts/plots/'
dataset_folder = '/home/elims/elimstuff/programming/projects/nn_digits/datasets/'
label_filename = dataset_folder + 'train-labels-idx1-ubyte'
image_filename = dataset_folder + 'train-images-idx3-ubyte'

### deal with label file
with open (label_filename, 'rb') as f:
    magic_label, n_labels = struct.unpack ('>II',f.read(8))
    labels = np.fromfile (f, dtype=np.uint8)
f.close ()

### deal with image file
with open (image_filename, 'rb') as f:
    magic_image, n_images, n_rows, n_cols = struct.unpack ('>IIII',f.read(16))
    images = np.fromfile (f, dtype=np.uint8)
f.close ()
### reshape `images` to a n_images x n_rows x n_cols matrix
### and normalize all pixel to 255
images = images.reshape (n_images, n_rows, n_cols) / 255.

### define some variables
n_pixels_per_axis  = n_rows  # number of pixels per dimension
n_samples = 100              # number of samples to be displayed;
                             # its sqrt must be an integer for now

### ramdonly select a small subset of data to visualize
n_samples_per_axis = int (np.sqrt (n_samples))
##  if python 3
#sampled_indices = np.array (random.choices (range (0, len (labels)-1), k=n_samples))
##  if python 2.7
sampled_indices = np.array (random.sample (range (0,len (labels)-1), k=n_samples))
sampled_images = images [sampled_indices]
sampled_labels = labels [sampled_indices]

### set up a n_samples x n_samples plot
h = plt.figure (figsize=(7.5, 5.5))

##  divide plot into sub plots
gs = gridspec.GridSpec (n_samples_per_axis, n_samples_per_axis)
gs.update (wspace=0.1, hspace=0.1)

##  each of x / y axis must have its number of bins = n_pixels_per_axis
edges = np.arange (0, n_pixels_per_axis+1)
edges = edges[:-1] + 0.5*(edges[1:] - edges[:-1])

##  loop through each selected image and plot
for index, image in enumerate (sampled_images):

    # location of the index-th subplot
    axis = h.add_subplot (gs[index])

    # inverse the content so it is not upside down
    content = image[::-1]

    # plot intensity
    axis.pcolormesh (edges, edges, content, vmin=0, vmax=1, cmap=plt.get_cmap ('gray'))

    # hide all axes ticks
    axis.get_xaxis ().set_ticks ([])
    axis.get_yaxis ().set_ticks ([])

### save plot
plt.suptitle (str (n_samples) + ' samples of training data')
h.savefig (plot_folder + 'training_samples.png')
plt.close ('all')
