import tensorflow as tf


class NeuralNet:
    def __init__(self, n_layers, n_neurons, lr, activation, loss, board_size):
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.lr = lr
        self.activation = activation
        self.loss = loss
        self.board_size = board_size