import tensorflow as tf

from . import nn_options


class BoardGameNet:
    def __init__(self,
                 n_layers=2,
                 n_neurons=20,
                 lr=0.001,
                 activation="relu",
                 output_activation="softmax",
                 loss="categorical_crossentropy",
                 optimizer="Adam",
                 board_size=6):
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.lr = lr
        self.activation = nn_options.activation_functions[activation]
        self.output_activation = nn_options.activation_functions[output_activation]
        self.loss = loss
        self.board_size = board_size
        self.optimizer = nn_options.optimizers[optimizer](
            learning_rate=self.lr)

        self.build_model()

    def build_model(self):
        self.model = tf.keras.models.Sequential()
        # Input needs to be the board size + 1 to include player ID
        self.model.add(tf.keras.layers.Dense(
            self.n_neurons, activation=self.activation, input_shape=(self.board_size**2 + 1,)))

        for _ in range(self.n_layers - 1):
            self.model.add(tf.keras.layers.Dense(
                self.n_neurons, activation=self.activation))

        self.model.add(tf.keras.layers.Dense(
            self.board_size**2, activation=self.output_activation))

        self.model.compile(optimizer=self.optimizer, loss=self.loss)
