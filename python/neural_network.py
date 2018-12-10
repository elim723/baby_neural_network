####
#### By Elim Thompson (12/03/2018)
####
#### Previously, we have built a NN and computed the cost function and its
#### derivatives w.r.t. all thetas. Now its time to minimize the cost to
#### find the best set of thetas to predict the digit of a given image.
############################################################################

### import packages
import numpy as np
import struct, random
from copy import deepcopy
from scipy.optimize import minimize

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
n_samples = n_images
images = images [:n_samples]
labels = labels [:n_samples]

### define some variables
n_pixels_per_axis  = n_rows             # number of pixels per dimension
n_pixels = n_rows * n_rows              # number of pixels per image
                                        # (i.e. number of input features / neurons)
n_classes = 10                          # number of classes 0-9
                                        # (i.e. number of output neurons)

### +-------------------------------------------------------------
### | Step 3. set up neural network structure
### +-------------------------------------------------------------
### set up neural network struction
n_neuron_per_hidden_layer = [25]
nn_struct = np.r_[n_pixels, n_neuron_per_hidden_layer, n_classes]

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

##  define lambda for regularization
Lambda = 1.0

def unroll_array (array):
    return np.concatenate ([array[i].ravel () for i in range (len (nn_struct)-1)]).ravel ()

def roll_array (array):
    nelements = (nn_struct[:-1]+1) * nn_struct [1:]
    nelements = np.r_[0, np.cumsum (nelements)]
    return np.array ([ array[nelements[i]:nelements[i+1]].reshape (nn_struct[i+1], nn_struct[i]+1)
            for i in range (len (nn_struct)-1) ])

flat_thetas = unroll_array (initial_thetas)
unflat_thetas = roll_array (flat_thetas)
unflat_thetas[0][-1][-10:]
initial_thetas[0][-1][-10:]

### +---------------------------------------------------------------
### | Step 5. compute cost function via a function
### +---------------------------------------------------------------
## Define a sigmoid function for an input matrix Z
def sigmoid (Z):
    return 1/(1+np.exp(-1*Z))

def compute_cost (thetas):

    ## define variables
    n_samples = len (labels)
    n_classes = 10
    Thetas = roll_array (thetas)

    ## Compute A_1, the neurons at the hidden layer
    A_0 = [images[i].flatten () for i in range (n_samples)]
    A_0 = np.vstack ([np.ones (n_samples), np.array (A_0).T])
    #  Calculate A_1 from A_0 and Theta[0]
    Z_1 = np.matmul (Thetas[0], A_0)
    A_1 = sigmoid (Z_1)
    #print ('A_0         shape: {0}'.format (A_0.shape))
    #print ('Theta_0     shape: {0}'.format (Thetas[0].shape))
    #print ('Z_1 = T0.A0 shape: {0}'.format (Z_1.shape))
    #print ('A_1 = g(Z1) shape: {0}'.format (A_1.shape))
    #print ('')

    ## Compute A_2, the neurons at the outer layer
    A_1 = np.vstack ([np.ones (n_samples), A_1])
    #  Calculate A_2 from A_1 and Theta[1]
    Z_2 = np.matmul (Thetas[1], A_1)
    A_2 = sigmoid (Z_2)
    #print ('A_1         shape: {0}'.format (A_1.shape))
    #print ('Theta_1     shape: {0}'.format (Thetas[1].shape))
    #print ('Z_2 = T1.A1 shape: {0}'.format (Z_2.shape))
    #print ('A_2 = g(Z2) shape: {0}'.format (A_2.shape))

    ## define a Y matrix representing the truth
    #  Initialize a zero matrix of the same size as A_2
    Y = np.zeros_like (A_2);
    for digit in range (n_classes):
        Y [digit, :] = labels==digit

    ## Compute the cost function
    #  Compute cost for each input image
    j = -Y * np.log (A_2) - (1-Y) * np.log (1-A_2)
    #  Averaged total cost is the sum of all costs
    J = np.sum (j) / float (n_samples)

    ## Compute the regularized cost function
    summed_squared = sum ([np.sum (Thetas[index][:,1:]**2) for index in range (len (Thetas))])
    extra = Lambda / 2 / float (n_samples) * summed_squared
    J += extra
    if not np.isfinite (J):
        J = 1e10
    print (J)
    return J

J = compute_cost (flat_thetas)
J

### +-----------------------------------------------------------
### | Step 6. compute derivative of cost function via backward propagation
### +-----------------------------------------------------------

##  Define the derivative of a sigmoid function
def dsigmoid (Z):
    return sigmoid (Z) * (1-sigmoid (Z))

def compute_derivatives (thetas):

    ## define variables
    n_samples = len (labels)
    n_classes = 10
    Thetas = roll_array (thetas)

    ## Compute A_1, the neurons at the hidden layer
    A_0 = [images[i].flatten () for i in range (n_samples)]
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
        Y [digit, :] = labels==digit

    ##  Initialize a list to hold all derivatives
    dJ_dThetas = []

    ##  Step 1: derivatives of J w.r.t. Theta_1
    ##          delta_2 at the outer layer
    delta_2 = A_2 - Y
    ##          dJ/dTheta_1
    dJ_dTheta1 = np.matmul (delta_2, A_1.T) / float (n_samples)
    ##          add the extra regularized term
    reg_term = deepcopy (Thetas[1])
    reg_term[:,0] = 0
    dJ_dTheta1 += Lambda / float (n_samples) * reg_term
    ##          append to dJ_dThetas
    dJ_dThetas.append (dJ_dTheta1)
    #print ('')
    #print ('A_2, Y         shape: {0}, {1}'.format (A_2.shape, Y.shape))
    #print ('delta_2 = A2-Y shape: {0}'.format (delta_2.shape))
    #print ('A_1            shape: {0}'.format (A_1.shape))
    #print ('dJ1 = d2.A1t   shape: {0}'.format (dJ_dTheta1.shape))

    ##  Step 2: derivatives of J w.r.t. Theta_0
    ##          delta_1 at the hidden layer
    delta_1 = np.matmul (Thetas[1][:,1:].T, delta_2) * dsigmoid (Z_1)
    ##          dJ/dTheta_0
    dJ_dTheta0 = np.matmul (delta_1, A_0.T) / float (n_samples)
    #print ('')
    #print ('delta_2              shape: {0}'.format (delta_2.shape))
    #print ('theta_1              shape: {0}'.format (Thetas[1].shape))
    #print ('A_1                  shape: {0}'.format (A_1.shape))
    #print ('delta1=T1t.d2*dg(A1) shape: {0}'.format (delta_1.shape))
    #print ('A_0                  shape: {0}'.format (A_0.shape))
    #print ('dJ0 = d1.A0t         shape: {0}'.format (dJ_dTheta0.shape))
    #print ('')
    #print ('delta_1[1:,:][0]: {0}'.format (delta_1[1:,:][0]))
    #print ('A_0.T[:,1]: {0}'.format (A_0.T[:,1]))
    #print (dJ_dTheta0[0][1])

    ##          add the extra regularized term
    reg_term = deepcopy (Thetas[0])
    reg_term[:,0] = 0
    dJ_dTheta0 += Lambda / float (n_samples) * reg_term
    ##          append to dJ_dThetas
    dJ_dThetas.append (dJ_dTheta0)

    ##  Step 3: reverse dJ_dThetas s.t its ordering follows Thetas
    dJ_dThetas = np.flip (np.array (dJ_dThetas), 0)

    return unroll_array (dJ_dThetas)

### +-----------------------------------------------------------
### | Step 8. sanity check
### +-----------------------------------------------------------
flat_thetas = unroll_array (initial_thetas)
unflat_thetas = roll_array (flat_thetas)

J = compute_cost (flat_thetas)
dJ = compute_derivatives (flat_thetas)

epsilon = 1e-4
for index in range (500):
    eps = np.zeros_like (flat_thetas)
    eps[index] = epsilon
    theta_plus = flat_thetas + eps
    theta_minus = flat_thetas - eps

    J_plus = compute_cost (theta_plus)
    J_minus = compute_cost (theta_minus)
    approx_dJ = (J_plus - J_minus)/(2*epsilon)
    #print ('{0} {1}'.format (dJ[index], approx_dJ))

### +-----------------------------------------------------------
### | Step 7. minimize cost function with provided derivatives
### +-----------------------------------------------------------
flat_thetas = unroll_array (initial_thetas)
results = minimize (compute_cost, flat_thetas,
                    method='L-BFGS-B',
                    jac=compute_derivatives,
                    options={'disp':True, 'maxiter':15000})

print (results)

### +-----------------------------------------------------------
### | Step 8. get predictions
### +-----------------------------------------------------------
def predict (thetas):

    ## define variables
    n_samples = len (labels)
    n_classes = 10
    Thetas = roll_array (thetas)

    ## Compute A_1, the neurons at the hidden layer
    A_0 = [images[i].flatten () for i in range (n_samples)]
    A_0 = np.vstack ([np.ones (n_samples), np.array (A_0).T])
    #  Calculate A_1 from A_0 and Theta[0]
    Z_1 = np.matmul (Thetas[0], A_0)
    A_1 = sigmoid (Z_1)

    ## Compute A_2, the neurons at the outer layer
    A_1 = np.vstack ([np.ones (n_samples), A_1])
    #  Calculate A_2 from A_1 and Theta[1]
    Z_2 = np.matmul (Thetas[1], A_1)
    A_2 = sigmoid (Z_2)

    ## find max per column as prediction
    return np.argmax (A2, axis=0)

predicted = predict (results.x)
print ('| predicted | truth |')
print ('+-----------+-------+')
for i in range (n_samples[:10]):
    print ('| {0:9} {1:5}'.format (predicted[i], labels[i]))
