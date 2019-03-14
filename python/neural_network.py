####
#### By Elim Thompson (12/09/2018)
####
#### Once the user specified the structure, this neural_network class is
#### set up to perform the followings:
####   -- initialize randomized thetas (i.e. weights)
####   -- compute the cost and its derivatives w.r.t. each theta
####      provided a set of thetas and training samples
####   -- find the best set of thetas based on the training set
####   -- compute accuracy of predictions based on the best fit thetas
############################################################################

### import packages
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize

default_epsilon = 0.12        # small factor for randomizing initial thetas
default_Lambda  = 1.0         # regularization factor
default_hidden_layers = [25]  # structure of NN inner layers

sigmoid = lambda Z: 1 / (1+np.exp (-1*Z))
dsigmoid = lambda Z: sigmoid (Z) * (1-sigmoid (Z))

class neural_network (object):

    ''' A class to perform neural network training. '''

    def __init__ (self):

        self._nn_struct = None
        self._n_pixels  = None
        self._hidden_layers = None
        self._n_classes = 10 ## from 0 to 9

    @property
    def hidden_layers (self):
        if self._hidden_layers is None:
            raise UnboundLocalError ('Structure of neural network has not defined.')
        return self._hidden_layers

    @hidden_layers.setter
    def hidden_layers (self, input):
        try:
            ### if input is a list or an array, make sure each
            ### element is an integer by converting them to int
            self._hidden_layers = [int (i) for i in input]
        except:
            try:
                ### if input is an integer / float, hidden layer is just
                ### 1 layer with number of neurons = the flor of input
                self._hidden_layers = [int (input)]
            except:
                message = 'Given input is type of {0}.'.format (type (input))
                message += 'List / array of numbers OR a number is expected.'
                raise ValueError (message)

    @property
    def n_pixels (self):
        if self._n_pixels is None:
            raise UnboundLocalError ('Input n_pixels has not defined.')
        return self._n_pixels

    @n_pixels.setter
    def n_pixels (self, input):
        try:
            ### if input is an integer or a float, n_pixels is the
            ### floor of the float
            self._n_pixels = int (input)
        except:
            message = 'Given input is type of {0}.'.format (type (input))
            message += 'A number is expected.'
            raise ValueError (message)

    @property
    def nn_struct (self):

        if self._hidden_layers is None:
            raise UnboundLocalError ('Hidden layers must be defined first.')
        if self._n_pixels is None:
            raise UnboundLocalError ('First layer (n_pixels) must be defined first.')

        return np.r_[self._n_pixels, self._hidden_layers,
                     self._n_classes]

    @property
    def n_layers (self):
        return len (self.nn_struct)

    @property
    def n_thetas_per_layer (self):
        return (self.nn_struct[:-1]+1) * self.nn_struct [1:]

    def initialize_thetas (self, epsilon=default_epsilon,
                           return_1d=False):

        ''' A public function to obtain a randomized values for all initial thetas
            as a 1D array. These initial values don't really matter, because
            they will be optimized later. Ideally, all thetas (or weights) should
            start from zeros. That is, we start with a flat line with zero slope.
            However, setting their initial values to absolutely zero would lead
            to degeneracy during minimization - the NN algorithm breaks down
            because all neurons in all layers will perform the same calculations.
            Therefore, in practice, we usually set them to be very close to zeros.
            Here, we start with some randomized initial thetas around zero with a
            perturbation of small epsilon.

            To avoid looping, this function initializes all thetas as a 1D array.
            Suppose there are M neurons in the 0-th layer, N neurons in the 1-st
            layer, and 10 neurons in the last layer. Between each layer has a matrix
            of thetas: Theta1 for the weights betwee the 0-th and 1-st layer, and
            Theta2 for the weights betwee the 1-st and 2-nd layer. Theta1 has a
            dimension of (M+1) x N, where the extra `+1` neuron corresponds to the
            constant (y-intercept) parameter of the fitted line. Similary, Theta2
            has a dimension of (N+1) x 10. This function generates an array of small
            random numbers with a total length equal to the number of thetas from
            both Thetas (i.e. (M+1) x N + (N+1) x 10). Once an array of random
            numbers are initialized, the generator function _roll_array () is used
            to return an array of matrices with correct shapes.

            input params
            ------------
            epsilon  (float): small factor for randomization
            return_1d (bool): If True, return a giant 1d array of all matrix
                              elements. Otherwise, return an array where each
                              element is a matrix.

            return params
            -------------
            random_array (np.ndarray): an array of matrices.
        '''

        ### generate an array of random numbers for all thetas
        random_thetas = np.random.rand (sum (self.n_thetas_per_layer))
        ### rescale to -epsilon to epsilon
        random_thetas = random_thetas * 2 * epsilon - epsilon

        ### return 1d array if asked
        if return_1d: return random_thetas

        ### roll the randomized thetas to an array of matrices with
        ### correct sizes
        random_array = list (self._roll_array (random_thetas))
        ### return a numpified array
        return np.array (random_array)

    def _roll_array (self, array):

        ''' This is a private generator function that put a 1D array into an array
            of matrices with the corresponding shape. The input array must have a
            length of sum_i [(m_i+1) x m_(i+1)] where m_i is the number of neurons
            of each layer from the input layer all the way to the last hidden layer
            (i.e. last end layer is excluded). This function then locates the indices
            in the 1D array where elements belong to the different layers and put
            them into the corresponding shapes.

            input params
            ------------
            array (np.ndarray): a 1D array that has a length equal to the total
                                number of thetas of the neural network.

            yield params
            -------------
            elements_at_ith_layer (np.ndarray): yield a matrix corresponding to a
                                                layer each time this function is
                                                called.
        '''

        ### make sure the length of input array is the same as the
        ### total number of thetas in this neural network
        assert (len (array) == np.sum (self.n_thetas_per_layer)), \
               "Input array does not have the same length as total number of thetas."

        ### identify the indices that locate elements of next layer
        indices = np.r_[0, np.cumsum (self.n_thetas_per_layer)]

        ### look through each layer (exclude last layer) and spit out
        ### the matrix of that layer
        for ith in range (self.n_layers-1):
            ## identify the thetas of this layer based on indices
            start_index, end_index = indices[ith], indices[ith+1]
            elements_at_ith_layer = array [start_index:end_index]
            ## determine the expected shape of this matrix
            shape_at_ith_layer = (self.nn_struct[ith+1], self.nn_struct[ith]+1)
            ## spit out matrix with expected shape
            yield elements_at_ith_layer.reshape (shape_at_ith_layer)

    def _unroll_array (self, array):

        ''' A private function that flattens an array of matrices into one giant
            1D array.

            input params
            ------------
            array (np.ndarray): an array of matrices

            return params
            -------------
            flattened_array (np.ndarray): a 1D array of all elements
        '''

        ### flatten each matrix in the input array
        ### each element is a 1D array from one matrix
        flattened_matrices = [array[i].ravel () for i in range (self.n_layers-1)]

        ### further flatten matrices to a 1D array with length
        ### equal to the total number of elements in the input array
        return np.concatenate (flattened_matrices).ravel ()

    def compute_cost (self, thetas, images, labels,
                      Lambda=default_Lambda, debug=False):

        ''' Give a set of images, this function calculates the cost between the
            predictions from the given set of thetas and the truth. For each image,
            a pixel value is the input of a neuron in the first layer. The value
            of each neuron at the second layer is then calculated by a linear
            combination of all first layer neurons and the corresponding set of
            thetas (or weights). This calculation is repeated for each layers until
            the last layer with 10 neurons. The value of each of these 10 neurons
            is the probability that this image belongs to this class between 0 and
            9. The cost of the input thetas measures how accurate the prediction is
            by comparing it to the truth given by the input labels. This calculation
            is repeated for all images, and the total cost of the given thetas is
            the mean cost from all images.

            To minimize the importance of each individual weights, regularization
            terms are added to the cost. By adding a small factor that depends on
            the square of theta, larger thetas would be penalized when minimizing
            the cost.

            To avoid memory-intensive for-loops, the cost calculation is performed
            via matrices using a generator function. Although these make it harder
            to understand the code, it significantly improves performances.

            input params
            ------------
            thetas (np.ndarray): a 1D giant array of flattened theta matrices
            images (np.ndarray): a n_sample x n_pixels_per_row x n_pixels_per_col
                                 matrix. Each pixel value is between 0 and 1 depending
                                 on the intensity of the pixel.
            labels (np.ndarray): a 1D array of the true value of the image
            Lambda (float)     : regularization factor
            debug  (bool)      : If True, print progress

            return params
            -------------
            j (float): mean cost of predictions from input thetas on input images
        '''

        ### define / initialize parameters
        n_samples        = len (labels)               # number of input samples
        theta_squared    = 0.                         # regularization term
        thetas_generator = self._roll_array (thetas)  # input thetas in matrix form

        ### start with input images at first layer
        ##  Each pixel in an image is the input of a neuron in the first layer. A
        ##  matrix `A` is defined such that each column corresponds to an input
        ##  image, and each row corresponds to the pixel values of the image.
        A = np.array ([images[i].ravel () for i in range (n_samples)]).T

        if debug:
            print ('| Computing the cost from {0} sample sizes ...'.format (n_samples))
            print ('| ')

        ### loop through each hidden layer to replace the value of A,
        ### which is the sigmoid function of the linear contributions.
        for ith in range (self.n_layers-1):

            ## Get the weights a.k.a. the thetas of this layer
            theta_i = next (thetas_generator)
            ##  Add an extra row of 1 for bias unit
            A = np.vstack ([np.ones (n_samples), A])

            if debug:
                print ('|  +-----------------------------------------------')
                print ('|  | At {0}-th hidden layer, '.format (ith))
                print ('|  |   -- A shape    : {0}'.format (A.shape))
                print ('|  |   -- theta shape: {0}'.format (theta_i.shape))

            ## At the i-th layer, a matrix `Z` is defined as the dot product
            ## between the theta of this layer and previous `A`. `Z` is the
            ## linear combination from all neurons of the i-th layer multiplied
            ## by the corresponding weight a.k.a. theta.
            Z = np.matmul (theta_i, A)
            ## Matrix `A` of this layer is simply the sigmoid of Z
            A = sigmoid (Z)

            if debug:
                print ('|  |   -- theta shape: {0}'.format (theta_i.shape))
                print ('|  |   -- A shape    : {0}'.format (A.shape))
            #print ('A_0         shape: {0}'.format (A_0.shape))
            #print ('Theta_0     shape: {0}'.format (Thetas[0].shape))
            #print ('Z_1 = T0.A0 shape: {0}'.format (Z_1.shape))
            #print ('A_1 = g(Z1) shape: {0}'.format (A_1.shape))
            #print ('')

            ## Accumulate regularization term. The first column which corresponds
            ## to the `constant` y-intercept parameters are ignored.
            theta_squared += sum ([np.sum (theta_i[:,1:]**2)])

        ### After the loop, the number of column in matrix `A` equals to the number
        ### of samples, and the number of rows is 10. The element in i-th column and
        ### j-th row corresponds to the propbability that the i-th image belongs to
        ### the j-th class, where j-th class can be any digits between 0 and 9.

        ### Now that we have the predictions `A`, we need to define
        ### a similar matrix of truth `Y`. This matrix has the same
        ### size as `A` but its elements are either 0 or 1 - an image
        ### must belong to one and only one of the 10 classes.
        Y = np.zeros_like (A)
        for digit in range (self._n_classes):
            Y [digit, :] = labels==digit

        ### To compare the prediction and truth, a cost is defined for each input
        ### image. Note that `*` here is performed element-wise and is not a dot
        ### product. Therefore, `J` is a matrix that has the same size as `A` and `Y`.
        J = -Y * np.log (A) - (1-Y) * np.log (1-A)
        ### The averaged total cost is the mean of all costs. This averaged cost
        ### is similar to a chi2 in a simple line fit.
        j = np.sum (J) / n_samples

        ### Finally, regularizatoin is added to the total cost. The purpose of
        ### regularization is to reduce the importance of each theta / weight by
        ### adding a small factor that depends on the square of theta. This is
        ### similar to adding penalties during chi2 minimization when theta is
        ### getting too large.
        j += Lambda / 2 / n_samples * theta_squared

        ### If the cost is infinite, set the cost to be a very large value.
        if not np.isfinite (j): j = 1e10

        return j

'''

J = compute_cost (flat_thetas)
J

### +-----------------------------------------------------------
### | Step 6. compute derivative of cost function via backward propagation
### +-----------------------------------------------------------


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
'''
