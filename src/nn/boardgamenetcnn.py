import numpy as np
import tensorflow as tf

from . import nn_options


class BoardGameNetCNN:
    def __init__(self,
                 n_layers=2,
                 n_neurons=20,
                 lr=0.001,
                 activation="relu",
                 output_activation="softmax",
                 loss="categorical_crossentropy",
                 optimizer="Adam",
                 board_size=6,
                 saved_model=None):
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.lr = lr
        self.activation = nn_options.activation_functions[activation]
        self.output_activation = nn_options.activation_functions[output_activation]
        self.loss = loss
        self.board_size = board_size
        self.optimizer = nn_options.optimizers[optimizer](
            learning_rate=self.lr)

        if not saved_model:
            self._build_model()
        else:
            self.model = tf.keras.models.load_model(saved_model)

    def _build_model(self):
        self.model = tf.keras.models.Sequential()
        # The input is a 2D array of size (board_size, board_size) with 5 channels
        # Channel 0 is player 1's cells, channel 2 is player 2's cells, channel 3 is empty cells,
        # channel 4 is 1 if current player is player 1, channel 5 is if current player is player 2
        input_shape = (self.board_size, self.board_size, 5)

        self.model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
        self.model.add(tf.keras.layers.Conv2D(
            32, kernel_size=(3, 3), padding='same', activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Flatten())

        for _ in range(self.n_layers):
            self.model.add(tf.keras.layers.Dense(
                self.n_neurons, activation=self.activation))

        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dense(
            self.board_size**2, activation=self.output_activation))

        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def fit(self, X, y, epochs=5, batch_size=32):
        self.model.fit(X, y, validation_split=0.2,
                       epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, X):
        prediction = self.model.predict(X, verbose=0)
        # Element wise multiplication to remove occupation of empty cells
        # All unoccupied cells are 1, thereby removing occupied cells
        mask = X[0, :, :, 2].flatten()

        prediction_occupied_removed = prediction * mask
        predictions_normalized = prediction_occupied_removed / \
            np.sum(prediction_occupied_removed)

        return predictions_normalized.reshape((self.board_size, self.board_size))

    def save_model(self, path):
        self.model.save(path)
