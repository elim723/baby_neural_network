import sys
sys.path.append ('python/')
from data_loader import data_loader
from neural_network import neural_network

data = data_loader ()
data.set_train_subset (100, random_sample=True)
print (len (data.train_labels))

nn = neural_network ()
nn.hidden_layers = 25
nn.n_pixels = data.n_pixels
print (nn.nn_struct)

initial_thetas = nn.initialize_thetas (return_1d=True)
initial_cost   = nn.compute_cost (initial_thetas, data.train_images,
                                  data.train_labels, debug=True)
initial_dcost  = nn.compute_dcost (initial_thetas, data.train_labels,
                                   debug=True)

nn.check_derivatives (initial_thetas, data.train_images,
                      data.train_labels, debug=True)

results = nn.fit (initial_thetas)
