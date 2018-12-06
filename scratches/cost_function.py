####
#### By Elim Thompson (12/02/2018)
####
#### This script focuses on computing the cost function of a
#### simple neural network structure. A neural network connects
#### the different features via sets of weights / thetas, each
#### of which indicates the importance or the contributions
#### from the corresponding features. A neural network of
#### a given structure is therefore defined by these thetas.
#### The goal to have a predictive neural network is to find
#### the values of these thetas that gave accurate predictions.
####
#### The accuracy of a neural network is described by the cost
#### function. It is simply a quantity that measures how
#### different the prediction is from its truth. This script
#### explains how a neural network is built, how such a cost
#### function is defined, and how to compute the cost of the
#### given sets of thetas.
####
#### Note that because we do not know the theta's yet, we
#### have some initialized thetas very close to zeros. It implies
#### that the predictive-ness of this model is very bad. To
#### find the numerical values of these thetas, we need to know
#### the derivatives of the cost function, which is explained
#### in cost_function_derivative.py. Once we have that set up
#### we can find the values of thetas that will give us the
#### best predictive power.
#####################################################################

### import packages
import numpy as np
import struct, random, math

### +------------------------------------------
### | Step 1. read training data
### +------------------------------------------
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

### +------------------------------------------
### | Step 2. massage data
### +------------------------------------------
### reshape `images` to a n_images x n_rows x n_cols matrix
### and normalize all pixel to 255
images = images.reshape (n_images, n_rows, n_cols) / 255.

### pick first 10 samples to build NN
n_samples = 10
sampled_images = images [:n_samples]
sampled_labels = labels [:n_samples]

### define some variables
n_pixels_per_axis  = n_rows             # number of pixels per dimension
n_pixels = n_rows * n_rows              # number of pixels per image
                                        # (i.e. number of input features / neurons)
n_classes = len (np.unique (labels))    # number of classes 0-9
                                        # (i.e. number of output neurons)

### +-------------------------------------------------------------
### | Step 3. set up neural network structure
### |
### | For simplicity, we have only three layers: an input layer,
### | a hidden layer, and an output layer. It will be generalized
### | later for more complicated structure.
### | See ../supplements/one_neuron_decoded.png for this NN
### | structure.
### |
### | Each layer has different number of neurons (or nodes.).
### | In general, the values of a set of neurons at a given layer
### | is denoted by a column vector `al`. Here, `a` is a column
### | vector with a size of n_neutrons x 1, and `l` represents
### | which layer the `a` refers to. For example, `a0` is a set
### | of neuron values at the 0th layer.
### |
### | When fitting the weights between layers, an extra neuron
### | is always added. This extra neuron is the bias unit that
### | is always set to 1; you can think of it as the y-interception,
### | a constant value, when fitting a line.
### |
### | 1. The 0th layer = an input layer
### |    Given each image has 28 x 28 = 784 pixels, this layer
### |    has a total of 784 + 1 neurons.
### |    In this project, `x` is the column vector representing
### |    the intensities from 784 pixels of a given training image.
### |    `X` denotes the matrix with a size of (784 + 1) x n_images.
### |    Since the input neurons are the same as data, a0 is either
### |    x or X depending on how many images you are fitting.
### |
### | 2. The 1st layer = the hidden layer
### |    Here, we set the NN only have 1 hidden layer with 25
### |    neurons. Again, a1 [25 x 1] is the column vector of the
### |    values of all neurons in this hidden layer.
### |
### | 3. The 2nd layer = the output layer
### |    The output layer is equal to the number of classes we
### |    we have. Because an image must belong to one of the 10
### |    digits bewteen 0 and 9, we have a total of 10 classes.
### |    Hence, we have 10 neurons at the output layers. For
### |    each image, it has 10 neuron values `a2`, each of which
### |    represents the probability that the image belongs to that
### |    class. The prediction is, therefore, the maximum of all
### |    elements in `a2`.
### +-------------------------------------------------------------
### set up neural network struction
n_neuron_per_hidden_layer = [25]
nn_struct = np.r_[n_pixels, n_neuron_per_hidden_layer, n_classes]
print (nn_struct) # [784, 25, 10]

### +-------------------------------------------------------------
### | Step 4. randomize initial thetas
### |
### | The most basic concept in a NN is the weights, or thetas in
### | this project, between two layers. Given three layers in this
### | project, we have two sets of thetas.
### |
### | For example, a set of thetas connects the 784 + 1 neurons
### | at the input layer and the 25 neurons in the hidden layer.
### | Let us focus on the first neuron in the hidden layer (i.e.
### | `a_11`, where the first `1` means the hidden layer, and
### | second `1` refers to the first neuron). The value of this
### | neuron is a sigmoid function of a variable `z_11`, which
### | is a linearly summed contribution from all input neurons.
### |
### | To be specific, the variable z_11 is given by
### |    z_11 = theta_010 * a_00 + theta_011 * a_01 + ...
### | Here, each theta_ijk denotes the i-th set of theta
### | contributing to the j-th neuron in the hidden layer from
### | the k-th neuron in the input layer. Note that, `a_00` is
### | the bias unit that is always set to 1, whereas theta_010
### | is the weight assigned to the bias unit.
### |
### | With the z_11, the value of `a_11` is therefore a sigmoid
### | function of z_11. The use of sigmoid function is to make
### | sure the output probability based on the input z
### | is between 0 and 1.
### |
### | To build a NN, one can simply repeat this process for
### | all neurons in all layers until the output layers. And the
### | prediction from this NN model is the maximum neuron value
### | at the last layer.
### |
### | For those who know linear algebra, you can easily see
### |     z_11 = theta_01 * a_0
### | where theta_01 is a 1 x (784 + 1) row vector, whereas a_0
### | is a (784 + 1) x 1 column vector, giving z_11 a scalar.
### | One can generalize this to get a matrix `Z_1`,
### |     Z_1 = Theta_0 * a_0
### | where Theta_0 is a 25 x (784 + 1) matrix, And, a_0 is still
### | the (784 + 1) x 1 column vector because all neurons at the
### | hidden layer uses the same input values; it is the weights
### | that are different between each neurons at the hidden layer.
### | Note that `Z_1` is a 25 x 1 column vector, its sigmoid
### | function `a_1` is also a 25 x 1 column vector, representing
### | the values at each neuron at the hidden layer.
### |
### | To generalize it even more to include, say, N images at
### | a time, we have A_0 = X, which is a (784 + 1) x N matrix.
### |     Z_1 = Theta_0 * A_0
### | where Theta_0 is still a 25 x (784 + 1) matrix. Here, `Z_1`
### | is a 25 x N matrix. Its sigmoid function `A_1` [25 x N]
### | represents the neuron values at each node for each image.
### |
### | What do the neuron values mean? If we repeat the same
### | calculation to the final output layer, the neuron values
### | are the probabilities that an input image belongs to the
### | output class. For example, the probabilities that this
### | image is 0/1/2/3/4/5/6/7/8/9 are
### | 0.01/0.02/0.01/0.99/0.01/0.02/0.01/0.01/0.02/0.02. So,
### | this NN model with the two sets of thetas predicts this
### | image belongs to a digit 3 class.
### |
### | Therefore, the thing that defines a neural network is the
### | values of thetas. We want to find the values of the two sets
### | of thetas that can give accurate prediction. In a math sense,
### | we want to find a set of thetas that minimizes the error
### | between the prediction and the truth.
### |
### | To do that we need to fist have some initial values of
### | thetas. The actual values don't really matter, because they
### | will be optimized later. But we usually set them to be very
### | close to zeros; that is, in two dimension, we start with a
### | flat straight line with zero slope. While setting them all
### | to zero is ideal, the NN algorithm will break down because
### | all neurons in all layers will perform the same calculations.
### | So, we start with some randomized initial thetas around
### | zero with a perturbation of small epsilon.
### +-------------------------------------------------------------
def rand_initial_thetas (n_rows, n_cols, epsilon):

    ### get a randomized matrix between 0 and 1
    random_matrix = np.random.rand (n_rows, n_cols)
    ### rescale to -epsilon to epsilon
    random_matrix = random_matrix * 2 * epsilon - epsilon
    ### return a numpified the randomized matrix
    return np.array (random_matrix)

##  initial thetas are randomized around zero to
##  break the symmetry
epsilon = 0.12
initial_thetas = np.array ([ rand_initial_thetas (nn_struct[layer]+1, nn_struct[layer+1], epsilon)
                             for layer in range (len (nn_struct)-1) ])
Thetas = initial_thetas

### +------------------------------------------------------------------------
### | Step 5. compute cost function
### |
### | SO, here is the fun part. As explained above, we want to find the set
### | of thetas that minimizers the error between the prediction and truth.
### | One can set up a function such as ~ (prediction - truth)**2 to be
### | minimized. This function is called the cost function because we want
### | to minimize the cost (i.e. the error) given by a set of thetas. Because
### | we are using sigmoid function, the corresponding cost function J is
### |
### |    J (theta) = 1/m sum_i sum_k
### |            {-y_i_k log [h(x_i)_k] - (1-y_i_k) log [1-h(x_i)_k]}
### |
### | Here, `y_i_k` is a dummy column vector that represents the true class.
### | For example, if an input image is a 5, its true vector transpose is
### | given by
### |    y.T = [0 0 0 0 0 1 0 0 0 0].
### | The h(x_i)_k is the k-th output neuron at the output layer. That is,
### | for each image, we compute its prediction for all classes; the cost
### | function then compares the predicted classes and the truth for all
### | input images. And the 1/m factor refers to the mean cost from all
### | images.
### |
### | The above cost function is not regularized. Regularization simply
### | means adding extra penalty terms in the cost function above. This
### | extra terms are theta**2 multiplied by a constant `lambda`.
### |
### |    J_reg (theta) = J_noReg + lambda / 2m *
### |                                 sum_i sum_j sum_k theta**2
### |
### | The purpose of these extra terms is to prevent overfitting. When
### | we have more than 700 neurons at one layer, the number of features
### | are too large compared to a training set of, say, 10 samples. When
### | minimizing the cost function, the minimizer would tend to make
### | thetas unrealistically large to fit the input images. By adding
### | these extra lambda * theta**2 terms penalizes those large thetas,
### | and the minimizer will very unlikely go to those unrealistic
### | regimes of thetas.
### |
### | Below shows how I compute the cost given a set of thetas.
### +------------------------------------------------------------------------

## Define a sigmoid function for an input matrix Z
def sigmoid (Z):
    return 1/(1+np.exp(-1*Z))

## +------------------------------------------------------
## | Step 1. Compute A_1, the neurons at the hidden layer
## +------------------------------------------------------
#  Massage input values A_0 = X
#  1. flatten each 28 x 28 image to one row vector with 784 elements
A_0 = [sampled_images[i].flatten () for i in range (n_samples)]
#  2. add an extra row of 1's for bias unit
A_0 = np.vstack ([np.ones (n_samples), np.array (A_0).T])
#  Check A_0 shape
print (A_0.shape) # (785, 10)

#  Calculate A_1 from A_0 and Theta[0]
#  1. compute Z_1, which is the linear contribution from all input
#     neurons at all hidden neurons. It has a dimension of 25 x 785.
#     To avoid for loop, we do a dot product between Thetas[0] and
#     the input values.
Z_1 = np.matmul (Thetas[0].T, A_0)
#  Check Z_1 shape
print (Z_1.shape) # (25, 10)
#  2. Obtain A_1, which are simply the sigmoid of Z_1
A_1 = sigmoid (Z_1)
print (A_1.shape) # (25, 10)

## +------------------------------------------------------
## | Step 2. Compute A_2, the neurons at the outer layer
## |         This A_2 are our final predictions.
## +------------------------------------------------------
#  Massage input values A_1
#  1. add an extra row of 1's for bias unit
A_1 = np.vstack ([np.ones (n_samples), A_1])
#  Check A_1 shape
print (A_1.shape) # (11, 25)

#  Calculate A_2 from A_1 and Theta[1]
#  1. compute Z_2, which is the linear contribution from all input
#     neurons at all hidden neurons. It has a dimension of 10 x 25.
#     To avoid for loop, we do a dot product between Thetas[1] and
#     the input A_1.
Z_2 = np.matmul (Thetas[1].T, A_1)
#  Check Z_2 shape
print (Z_2.shape)   # (10, 10)
#  2. Obtain A_2, which are simply the sigmoid of Z_2
A_2 = sigmoid (Z_2)
print (A_2.shape)   # (10, 10)

## +-----------------------------------------------------------
## | Step 3. Check predictions
## |         Given a set of thetas very close to zeros,
## |         this NN is expected to not have much predicting
## |         power. Indeed, this set of thetas say the 10
## |         sampled images can be any digits. That is, the
## |         probabilities are all ~ 50%.
## +-----------------------------------------------------------
print (np.round (A_2, 2))
# Here, each column = predictions from each of the 10 images.
# Each row = probability that this image belongs to a class.

#        1st   2nd   3rd   4th   5th   6th   7th   8th   9th  10th
# 0: [[ 0.58  0.59  0.59  0.59  0.58  0.58  0.57  0.59  0.57  0.6 ]
# 1:  [ 0.47  0.45  0.45  0.44  0.46  0.47  0.46  0.46  0.45  0.44]
# 2:  [ 0.53  0.54  0.53  0.53  0.54  0.53  0.53  0.52  0.54  0.52]
# 3:  [ 0.52  0.52  0.52  0.52  0.53  0.53  0.53  0.53  0.53  0.52]
# 4:  [ 0.59  0.58  0.57  0.59  0.57  0.57  0.59  0.58  0.58  0.58]
# 5:  [ 0.47  0.45  0.48  0.48  0.48  0.47  0.48  0.47  0.47  0.48]
# 6:  [ 0.52  0.53  0.51  0.5   0.52  0.52  0.51  0.52  0.51  0.52]
# 7:  [ 0.53  0.52  0.52  0.53  0.52  0.51  0.55  0.51  0.54  0.51]
# 8:  [ 0.45  0.46  0.45  0.44  0.45  0.45  0.45  0.45  0.46  0.45]
# 9:  [ 0.49  0.5   0.51  0.51  0.49  0.51  0.5   0.51  0.5   0.51]]

## +--------------------------------------------------------------
## | Step 4: define a Y matrix representing the truth
## |         To calculate the cost, we need to convert the y_i_k
## |         into a dummy binary matrix. It has the same dimesion
## |         as the prediction A_2
## +--------------------------------------------------------------
#  Initialize a zero matrix of the same size as A_2
Y = np.zeros_like (A_2);
Y[0,:]
#  For each column, the row index is turned to 1 if that index
#  equal to the digit.
for digit in range (n_classes):
    Y [digit, :] = sampled_labels==digit

print (sampled_labels) # [5 0 4 1 9 2 1 3 1 4]
print (Y)
#      1st 2nd 3rd 4th 5th 6th 7th 8th 9th 10th
# 0: [[ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
# 1:  [ 0.  0.  0.  1.  0.  0.  1.  0.  1.  0.]
# 2:  [ 0.  0.  0.  0.  0.  1.  0.  0.  0.  0.]
# 3:  [ 0.  0.  0.  0.  0.  0.  0.  1.  0.  0.]
# 4:  [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  1.]
# 5:  [ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
# 6:  [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
# 7:  [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
# 8:  [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
# 9:  [ 0.  0.  0.  0.  1.  0.  0.  0.  0.  0.]]

## +--------------------------------------------------------------
## | Step 5: compute the cost function
## |
## |  Per element calculation:
## |    J (theta) = 1/m sum_i sum_k
## |            {-y_i_k log [h(x_i)_k] - (1-y_i_k) log [1-h(x_i)_k]}
## |
## |  In terms of linear algebra
## |    J (theta) = 1/m sum_i [-Y * log (A_2) - (1-Y) * log (1-A_2)]
## +--------------------------------------------------------------
#  Compute cost for each input image
j = -Y * np.log (A_2) - (1-Y) * np.log (1-A_2)
#  Look at the costs per prediction per image.
#  The costs are pretty large and pretty even across all classes.
#  Once we find the right set of thetas, we should see these
#  costs to be << 1.
print (np.round (j, 2))
#        1st   2nd   3rd   4th   5th   6th   7th   8th   9th  10th
# 0: [[ 7.44,  7.46,  7.44,  7.46,  7.42,  7.38,  7.47,  7.4 ,  7.47, 7.48],
# 1:  [ 7.33,  7.24,  7.31,  7.38,  7.28,  7.24,  7.34,  7.25,  7.29, 7.3 ],
# 2:  [ 7.42,  7.45,  7.32,  7.32,  7.33,  7.38,  7.38,  7.35,  7.38, 7.35],
# 3:  [ 7.18,  7.19,  7.17,  7.12,  7.17,  7.23,  7.1 ,  7.2 ,  7.11, 7.22],
# 4:  [ 7.22,  7.12,  7.07,  7.08,  7.14,  7.08,  7.15,  7.15,  7.11, 7.13],
# 5:  [ 6.97,  6.89,  6.9 ,  6.85,  6.94,  6.93,  7.01,  6.89,  7.01, 6.85],
# 6:  [ 7.31,  7.27,  7.25,  7.23,  7.26,  7.25,  7.29,  7.25,  7.27, 7.26],
# 7:  [ 7.31,  7.27,  7.25,  7.23,  7.26,  7.25,  7.29,  7.25,  7.27, 7.26],
# 8:  [ 7.31,  7.27,  7.25,  7.23,  7.26,  7.25,  7.29,  7.25,  7.27, 7.26],
# 9:  [ 6.93,  6.93,  6.96,  6.88,  6.97,  6.96,  6.94,  6.94,  6.96, 6.92]])

# Averaged total cost is the sum of all costs
J = np.sum (j) / n_samples
print (J) # 72.0550976935

## +--------------------------------------------------------------
## | Step 6: compute the regularized cost function
## |
## | Per element calculation:
## |     J_reg (theta) = J_noReg + lambda / 2m *
## |                              sum_i sum_j sum_k theta**2
## |
## | In terms of linear algebra
## |     J_reg (theta) = J_noReg + lambda / 2m * np.sum (Theta**2)
## +--------------------------------------------------------------
#  Define some lambda
Lambda = 1.0
#  Compute the extra term
summed_squared = sum ([np.sum (Thetas[index]**2) for index in range (len (Thetas))])
extra = Lambda / 2 / n_samples * summed_squared
print (extra) # 4.77698645582
#  Compute regularized cost
J += extra
print (J) # 76.8320841493
