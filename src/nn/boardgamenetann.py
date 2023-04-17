import numpy as np
import tensorflow as tf

from . import nn_options


class BoardGameNetANN:
    def __init__(self,
                 neural_network_dimensions=(64, 32),
                 lr=0.001,
                 activation="relu",
                 output_activation="softmax",
                 loss="categorical_crossentropy",
                 optimizer="Adam",
                 board_size=6,
                 saved_model=None):
        self.neural_network_dimensions = neural_network_dimensions
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
        """Builds the neural network model.
        """
        self.model = tf.keras.models.Sequential()
        input_shape = (self.board_size**2 + 1,)

        self.model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

        for i in range(len(self.neural_network_dimensions)):
            self.model.add(tf.keras.layers.Dense(
                self.neural_network_dimensions[i], activation=self.activation))

        self.model.add(tf.keras.layers.Dense(
            self.board_size**2, activation=self.output_activation))

        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def fit(self, X, y, epochs=10, batch_size=32):
        """Fits the model.

        Args:
            X (np.ndarray): the training data (game states).
            y (np.ndarray): the target distribution.
            epochs (int, optional): number of epochs in training. Defaults to 10.
            batch_size (int, optional): the batch size. Defaults to 32.
        """
        self.model.fit(X, y, validation_split=0.2,
                       epochs=epochs, batch_size=batch_size)

    def predict_best_move(self, state=None, model_input=None):
        """Predicts the best move given the model input.

        Args:
            state (StateManager, optional): the state of the game. Defaults to None.
            model_input (np.ndarray, optional): the raw model input. Defaults to None.

        Returns:
            tuple[int, int]: the move to choose
        """
        nn_input = self.convert_to_nn_input(
            state) if model_input is None else model_input
        predictions = self.call(nn_input)

        prediction = np.argmax(predictions)
        move = (prediction // self.board_size, prediction % self.board_size)

        return move

    def predict_probabilistic_move(self, state=None, model_input=None):
        """Predicts the move according to the probability distribution given by the model.

        Args:
            state (StateManager, optional): the state of the game. Defaults to None.
            model_input (np.ndarray, optional): the raw model input. Defaults to None.

        Returns:
            tuple[int, int]: the move to choose
        """
        nn_input = self.convert_to_nn_input(
            state) if model_input is None else model_input

        moves = self.call(nn_input).reshape(-1,)
        indices = np.arange(len(moves))

        move = np.random.choice(indices, p=moves)
        move = (move // self.board_size, move % self.board_size)

        return move

    def call(self, X):
        """Predicts the output of the neural network given the input.
        Uses the __call__ method of the model, which is faster than using the predict method.

        Args:
            X (np.ndarray): the input to the neural network

        Returns:
            np.ndarray: the predictions for each cell
        """
        mask = (X[0, 1:] == 0).astype(int)
        # Convert to tensor
        X = tf.convert_to_tensor(X)
        prediction = self.model(X)

        # Convert output to numpy array
        prediction = prediction.numpy()

        prediction_occupied_removed = prediction * mask

        sum_prediction = np.sum(prediction_occupied_removed)
        # If the sum of the prediction is zero, then the mask is used as a fallback
        # to still return a valid move. This can happen when the model predicts something
        # to be zero.
        if sum_prediction == 0:
            prediction_occupied_removed = mask
            predictions_normalized = prediction_occupied_removed / \
                np.sum(prediction_occupied_removed)
        else:
            predictions_normalized = prediction_occupied_removed / sum_prediction
        return predictions_normalized.reshape((self.board_size, self.board_size))

    # Inspired by the article here: https://www.idi.ntnu.no/emner/it3105/materials/neural/gao-2017.pdf
    # Should make it possible to feed to convolutional neural network with 5 channels, 3 for occupancy
    # and 2 for each player's turn
    def convert_to_nn_input(self, state):
        """Converts the game state to the input format of the convolutional neural network.
        The 5 channels represent the game state as bitboards:
        Channel 1 is the cells occupied by player 1.
        Channel 2 is the cells occupied by player 2.
        Channel 3 is the cells currently unoccupied.
        Channel 4 are all 1's if the current player is 1.
        Channel 5 are all 1's if the current player is 2.

        Returns:
            np.ndarray: the nn input of shape (0, board_size, board_size, 5),
        """
        board = state.board
        player = state.player

        nn_input = np.zeros((self.board_size**2 + 1),)

        occupation = np.vectorize(lambda x: 1 if x.occupant == (
            1, 0) else 2 if x.occupant == (0, 1) else 0)

        nn_input[1:] = occupation(board).flatten()
        nn_input[0] = 1 if player == (1, 0) else 2 if player == (0, 1) else 0

        nn_input = np.expand_dims(nn_input, axis=0)

        return nn_input

    def save_model(self, path):
        self.model.save(path)
