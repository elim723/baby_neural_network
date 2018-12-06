####
#### By Elim Thompson (12/03/2018)
####
#### This script focuses on setting up the algorithm for computing
#### the derivatives of the cost function. Recall that, our goal is
#### to find the numerical values of thetas that minimizes the cost
#### function. In calculus, the most basic concept is that a function
#### is minimium when its derivative is equal to zero. Although the
#### cost function will be minimized using a minimizer, providing
#### the minimizer with the derivative of cost function can help
#### the minimizer to find the minimum much quicker!
####
#### Since the cost function is complicated, its derivatives with
#### respect to each of the thetas are even more complicated. One
#### trick to simplify the calculation is the backward propagation.
#### While the cost function is set up from the input layer all the
#### way to the output layer, its derivatives are set up from the
#### output layer back to the input layer. The basic idea is to ask,
#### for each node, how different the values of the neuron is from
#### the truth.
############################################################################

### import packages
import numpy as np
import struct, random
from copy import deepcopy

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

### pick first 11 samples to build NN
n_samples = 11
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
### +-------------------------------------------------------------
### set up neural network struction
n_neuron_per_hidden_layer = [25]
nn_struct = np.r_[n_pixels, n_neuron_per_hidden_layer, n_classes]
print (nn_struct) # [784, 25, 10]

### +-------------------------------------------------------------
### | Step 4. randomize initial thetas
### +-------------------------------------------------------------
def rand_initial_thetas (n_cols, n_rows, epsilon):
    ### get a randomized matrix between 0 and 1
    random_matrix = np.random.rand (n_rows, n_cols)
    ### rescale to -epsilon to epsilon
    random_matrix = random_matrix * 2 * epsilon - epsilon
    ### return a numpified the randomized matrix
    return np.array (random_matrix)

##  initial thetas are randomized around zero to break the symmetry
epsilon = 0.12
initial_thetas = np.array ([ rand_initial_thetas (nn_struct[layer]+1, nn_struct[layer+1], epsilon)
                             for layer in range (len (nn_struct)-1) ])
Thetas = initial_thetas

### +---------------------------------------------------------------
### | Step 5. compute cost function
### +---------------------------------------------------------------
## Define a sigmoid function for an input matrix Z
def sigmoid (Z):
    return 1/(1+np.exp(-1*Z))

## Compute A_1, the neurons at the hidden layer
A_0 = [sampled_images[i].flatten () for i in range (n_samples)]
A_0 = np.vstack ([np.ones (n_samples), np.array (A_0).T])
#  Calculate A_1 from A_0 and Theta[0]
Z_1 = np.matmul (Thetas[0], A_0)
A_1 = sigmoid (Z_1)

## Compute A_2, the neurons at the outer layer
A_1 = np.vstack ([np.ones (n_samples), A_1])
#  Calculate A_2 from A_1 and Theta[1]
Z_2 = np.matmul (Thetas[1], A_1)
A_2 = sigmoid (Z_2)

## define a Y matrix representing the truth
#  Initialize a zero matrix of the same size as A_2
Y = np.zeros_like (A_2);
for digit in range (n_classes):
    Y [digit, :] = sampled_labels==digit

## Compute the cost function
#  Compute cost for each input image
j = -Y * np.log (A_2) - (1-Y) * np.log (1-A_2)
#  Averaged total cost is the sum of all costs
J = np.sum (j) / n_samples

## Compute the regularized cost function
Lambda = 1.0
summed_squared = sum ([np.sum (Thetas[index]**2) for index in range (len (Thetas))])
extra = Lambda / 2 / n_samples * summed_squared
J += extra

### +-----------------------------------------------------------
### | Step 6. compute derivative of cost function via backward
### |         propagation
### |
### | The overall idea is to loop through each layer from the
### | outer most layer. For the ith layer, we need to compute the
### | delta_i. At the outer most layer, delta_2 is simply the
### | subtraction between prediction and truth. At the hidden
### | layer, the delta_1 is given by
### |     delta_i = Theta_i.T . delta_(i+1) * g' (Z_i).
### | Here, Theta_i are calculated previously from forward
### | propagation, delta_(i+1) is the error from the next
### | layer, and g' is the derivative of the sigmoid function.
### | Note that the `.` is the dot product between Theta_i.T
### | and delta_(i+1), where the bias units are taken out
### | from the Theta_i matrix. The `*` is an element-wise
### | multiplication between g' and the output from the
### | dot product.
### |
### | At the ith layer, after its delta is computed, the
### | derivatives of cost function with respect to the
### | Theta's are given by
### |     dJ/dTheta_i = delta_(i+1) . a_i.T / m
### | where delta_(i+1) is the delta from the next layer,
### | a_i is the sigmoid function of Z_(i-1), and m is
### | the number of traning samples. Note the `.` here
### | is a dot product, and the derivatives are partial
### | instead of total. Also note that dJ/dTheta_i is a
### | matrix with the same size as Theta_i.
### +-----------------------------------------------------------

##  Define the derivative of a sigmoid function
def dsigmoid (Z):
    return sigmoid (Z) * (1-sigmoid (Z))

##  Initialize a list to hold all derivatives
dJ_dThetas = []

##  +--------------------------------------------------------
##  | Step 1: derivatives of J w.r.t. Theta_1
##  +--------------------------------------------------------

##  First: delta_2 at the outer layer
##  At the outer layer, the error is simply the difference
##  between the prediction and the truth. Since we already
##  have the matrix Y and A_2, delta_2 is a simple subtraction
delta_2 = A_2 - Y
#   The differences are quite large ~ 0.5.
print (np.round (delta_2, 2))
#        1st   2nd   3rd   4th   5th   6th   7th   8th   9th  10th  11th
# 0: [[ 0.57 -0.44  0.56  0.57  0.56  0.56  0.58  0.54  0.57  0.57  0.56]
# 1:  [ 0.45  0.44  0.45 -0.55  0.44  0.43 -0.54  0.46 -0.54  0.46  0.45]
# 2:  [ 0.5   0.51  0.51  0.51  0.51 -0.49  0.5   0.51  0.5   0.5   0.52]
# 3:  [ 0.42  0.41  0.43  0.43  0.43  0.42  0.41 -0.57  0.42  0.43 -0.59]
# 4:  [ 0.56  0.55 -0.44  0.54  0.54  0.54  0.54  0.57  0.53 -0.45  0.55]
# 5:  [-0.48  0.51  0.54  0.53  0.54  0.51  0.53  0.54  0.53  0.53  0.53]
# 6:  [ 0.5   0.49  0.48  0.46  0.46  0.47  0.49  0.49  0.48  0.49  0.48]
# 7:  [ 0.58  0.6   0.56  0.56  0.56  0.59  0.58  0.56  0.58  0.57  0.58]
# 8:  [ 0.58  0.58  0.58  0.58  0.58  0.58  0.58  0.58  0.58  0.58  0.59]
# 9:  [ 0.61  0.6   0.59  0.59 -0.4   0.6   0.6   0.61  0.59  0.58  0.59]]

##  Second: dJ/dTheta_1
##  Its derivatives are given by
##     dJ/dTheta_1 = delta_2 . A_1.T / m
##  Note that because dJ/dTheta_1 is the same dimension
##  as Theta_1, we need to transpose the output before
##  appending to the dJ_dThetas list. And also note the
##  order of dJ_dThetas list is reversed order; this will
##  be fixed later.
dJ_dTheta1 = np.matmul (delta_2, A_1.T) / n_samples
print (dJ_dTheta1.shape)    # (10, 26)

##  Third: add the extra regularized term
reg_term = deepcopy (Thetas[1])
reg_term[:,0] = 0

dJ_dTheta1 += Lambda / n_samples * reg_term

##  Forth: append to dJ_dThetas
dJ_dThetas.append (dJ_dTheta1)
print (dJ_dThetas[0].shape) # (10, 26)
print (Thetas[1].shape)     # (10, 26)

##  +--------------------------------------------------------
##  | Step 2: derivatives of J w.r.t. Theta_0
##  +--------------------------------------------------------

##  First: delta_1 at the hidden layer
##  At the hidden layer, the delta_1 is given by
##         delta_1 = theta_1.T . delta_2 * g' (Z_0)
##  where `.` refers to dot product and `*` means element-wise
##  multiplication. Note that since theta_1 includes the bias
##  unit that should not be propagated to the first layer,
##  theta_1 in this equation should be theta_1[1:] instead.
delta_1 = np.matmul (Thetas[1][:,1:].T, delta_2) * dsigmoid (A_1[1:])
print (delta_1.shape) # (25, 11)

##  Second: dJ/dTheta_0
##  Its derivatives are given by
##     dJ/dTheta_0 = delta_1 . A_0.T / m
dJ_dTheta0 = np.matmul (delta_1, A_0.T) / n_samples
print (dJ_dTheta0.shape) # (25, 785)

##  Third: add the extra regularized term
reg_term = deepcopy (Thetas[0])
reg_term[:,0] = 0

dJ_dTheta0 += Lambda / n_samples * reg_term

##  Forth: append to dJ_dThetas
dJ_dThetas.append (dJ_dTheta0)
print (dJ_dThetas[1].shape) # (25, 785)
print (Thetas[0].shape)     # (25, 785)

##  +--------------------------------------------------------
##  | Step 3: reverse dJ_dThetas s.t its ordering follows Thetas
##  +--------------------------------------------------------

dJ_dThetas = np.flip (dJ_dThetas, 0)
print (dJ_dThetas[0].shape) # (25, 785)
print (Thetas[0].shape)     # (25, 785)
print (dJ_dThetas[1].shape) # (10, 26)
print (Thetas[1].shape)     # (10, 26)
