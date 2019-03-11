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
        self._n_neurons_at_end_layer = 10 ## from 0 to 9

    @property
    def hidden_layers (self):
        if self._hidden_layers is None:
            raise UnboundLocalError ('Structure of neural network has not defined.')
        return self._hidden_layers

    @hidden_layers.setter
    def hidden_layers (self, input):
        try:
            ## if input is a list or an array, make sure each
            ## element is an integer by converting them to int
            self._hidden_layers = [int (i) for i in input]
        except:
            try:
                ## if input is an integer / float, hidden layer is
                ## just 1 layer with number of neurons = the flor of input
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
            ## if input is an integer or a float, n_pixels is the floor
            ## of the float
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
                     self._n_neurons_at_end_layer]

    @property
    def n_layers (self):
        return len (self.nn_struct)

    @property
    def n_thetas_per_layer (self):
        return (self.nn_struct[:-1]+1) * self.nn_struct [1:]

    def _initialize_thetas (self, epsilon=default_epsilon):

        ''' A private function to obtain a randomized values for all initial thetas
            as a 1D array. These initial values don't really matter, because
            they will be optimized later. Ideally, all thetas (or weights) should
            start from zeros. That is, we start with a flat line with zero slope.
            However, setting their initial values to absolutely zero would lead
            to degeneracy during minimization - the NN algorithm breaks down
            because all neurons in all layers will perform the same calculations.
            Therefore, in practice, we usually set them to be very close to zeros.
            Here, we start with some randomized initial thetas around zero with a
            perturbation of small epsilon.

            Note that this function only initialize all thetas as a 1D array.
            Suppose there are M neurons in the 0-th layer, N neurons in the 1-st
            layer, and 10 neurons in the last layer. Between each layer has a matrix
            of thetas: Theta1 for the weights betwee the 0-th and 1-st layer, and
            Theta2 for the weights betwee the 1-st and 2-nd layer. Theta1 has a
            dimension of (M+1) x N, where the extra `+1` neuron corresponds to the
            constant (y-intercept) parameter of the fitted line. Similary, Theta2
            has a dimension of (N+1) x 10. This function generates an array of small
            random numbers with a total length equal to the number of thetas from
            both Thetas (i.e. (M+1) x N + (N+1) x 10). To roll the initialized
            thetas back to matrix-form, use the generator function _roll_array ().

            input params
            ------------
            epsilon (float): small factor for randomization

            return params
            -------------
            random_array (np.ndarray): flattened array of a (m+1) x n matrix
        '''

        ### generate an array of random numbers for all thetas
        random_array = np.random.rand (sum (self.n_thetas_per_layer))
        ### rescale to -epsilon to epsilon
        random_array = random_array * 2 * epsilon - epsilon
        ### return a numpified the randomized array
        return np.array (random_array)

    def _roll_array (self, array):

        ## the corresponding indices that locates thetas at next layer
        theta_indices = np.r_[0, np.cumsum (self.n_thetas_per_layer)]

        for ith in range (self.n_layers-1):
            start_index, end_index = theta_indices[ith], theta_indices[ith+1]
            elements_at_ith_layer = array [start_index:end_index]
            shape_at_ith_layer = (self.nn_struct[ith+1], self.nn_struct[ith]+1)
            yield elements_at_ith_layer.reshape (shape_at_ith_layer)



'''

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
'''
