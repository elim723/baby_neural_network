from data_loader import data_loader
from neural_network import neural_network

data = data_loader ()
data.set_train_subset (100, random_sample=True)

nn = neural_network ()
nn.n_pixels = data.n_pixels
nn.hidden_layers = [25]
