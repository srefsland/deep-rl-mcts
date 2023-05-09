import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from . import nn_options


class BoardGameNetANN:
    def __init__(
        self,
        neural_network_dimensions=(64, 32),
        lr=0.001,
        activation="relu",
        output_activation_actor="softmax",
        output_activation_critic="tanh",
        loss_actor="categorical_crossentropy",
        loss_critic="mse",
        optimizer="Adam",
        board_size=6,
        saved_model=None,
    ):
        self.neural_network_dimensions = neural_network_dimensions
        self.lr = lr
        self.activation = nn_options.activation_functions[activation]
        self.output_activation_actor = nn_options.activation_functions[
            output_activation_actor
        ]
        self.output_activation_critic = nn_options.activation_functions[
            output_activation_critic
        ]
        self.loss_actor = loss_actor
        self.loss_critic = loss_critic
        self.board_size = board_size
        self.optimizer = nn_options.optimizers[optimizer](learning_rate=self.lr)

        if not saved_model:
            self._build_model()
        else:
            self.model = tf.keras.models.load_model(saved_model)

        self.losses_actor = []
        self.val_losses_actor = []
        self.losses_critic = []
        self.val_losses_critic = []

        self.plot_created = False

    def _build_model(self):
        """Builds the neural network model."""
        input_layer = tf.keras.layers.Input(shape=(self.board_size**2 + 1,))
        hidden_layer = input_layer

        for i in range(len(self.neural_network_dimensions)):
            hidden_layer = tf.keras.layers.Dense(
                self.neural_network_dimensions[i], activation=self.activation
            )(hidden_layer)

        output_actor = tf.keras.layers.Dense(
            self.board_size**2,
            activation=self.output_activation_actor,
            name="actor",
        )(hidden_layer)

        output_critic = tf.keras.layers.Dense(
            units=1,
            activation=self.output_activation_critic,
            name="critic",
        )(hidden_layer)

        model = tf.keras.models.Model(
            inputs=input_layer, outputs=[output_actor, output_critic]
        )
        model.compile(
            optimizer=self.optimizer, loss=[self.loss_actor, self.loss_critic]
        )
        self.model = model

    def fit(self, X, y_actor, y_critic, epochs=10, batch_size=32):
        """Fits the model.

        Args:
            X (np.ndarray): the training data (game states).
            y_actor (np.ndarray): the target distribution for the actor network.
            y_critic (np.ndarray): the target distribution for the critic network.
            epochs (int, optional): number of epochs in training. Defaults to 10.
            batch_size (int, optional): the batch size. Defaults to 32.
        """
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)

        y = [y_actor, y_critic]
        losses = self.model.fit(
            X,
            y,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
        )

        self.losses_actor.append(losses.history["actor_loss"][-1])
        self.losses_critic.append(losses.history["critic_loss"][-1])
        self.val_losses_actor.append(losses.history["val_actor_loss"][-1])
        self.val_losses_critic.append(losses.history["val_critic_loss"][-1])

    def predict_best_move(self, state=None, player=None, model_input=None):
        """Predicts the best move given the model input.

        Args:
            state (StateManager, optional): the state of the game. Defaults to None.
            model_input (np.ndarray, optional): the raw model input. Defaults to None.

        Returns:
            tuple[int, int]: the move to choose
        """
        nn_input = (
            self.convert_to_nn_input(state, player)
            if model_input is None
            else model_input
        )
        predictions = self.call_actor(nn_input)

        prediction = np.argmax(predictions)
        move = (prediction // self.board_size, prediction % self.board_size)

        return move

    def predict_probabilistic_move(self, state=None, player=None, model_input=None):
        """Predicts the move according to the probability distribution given by the model.

        Args:
            state (StateManager, optional): the state of the game. Defaults to None.
            model_input (np.ndarray, optional): the raw model input. Defaults to None.

        Returns:
            tuple[int, int]: the move to choose
        """
        nn_input = (
            self.convert_to_nn_input(state, player)
            if model_input is None
            else model_input
        )

        moves = self.call_actor(nn_input).reshape(
            -1,
        )
        indices = np.arange(len(moves))

        move = np.random.choice(indices, p=moves)
        move = (move // self.board_size, move % self.board_size)

        return move

    def call_actor(self, X):
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
        prediction = self.model(X)[0]

        # Convert output to numpy array
        prediction = prediction.numpy()

        prediction_occupied_removed = prediction * mask

        sum_prediction = np.sum(prediction_occupied_removed)
        # If the sum of the prediction is zero, then the mask is used as a fallback
        # to still return a valid move. This can happen when the model predicts something
        # to be zero.
        if sum_prediction == 0:
            prediction_occupied_removed = mask
            predictions_normalized = prediction_occupied_removed / np.sum(
                prediction_occupied_removed
            )
        else:
            predictions_normalized = prediction_occupied_removed / sum_prediction
        return predictions_normalized.reshape((self.board_size, self.board_size))

    def call_critic(self, state, player):
        X = self.convert_to_nn_input(state, player)
        X = tf.convert_to_tensor(X)

        prediction = self.model(X)[1]

        prediction = prediction.numpy()

        return np.squeeze(prediction)

    # Inspired by the article here: https://www.idi.ntnu.no/emner/it3105/materials/neural/gao-2017.pdf
    # Should make it possible to feed to convolutional neural network with 5 channels, 3 for occupancy
    # and 2 for each player's turn
    def convert_to_nn_input(self, state, player):
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
        nn_input = np.zeros(
            (self.board_size**2 + 1),
        )

        nn_input[1:] = state.flatten()
        nn_input[0] = player

        nn_input = np.expand_dims(nn_input, axis=0)

        return nn_input

    def save_model(self, path):
        """Saves the model to the specified path.

        Args:
            path (str): path to save the model to.
        """
        self.model.save(path)

    def save_losses(self):
        """Saves the plot the losses to file."""
        if not self.plot_created:
            self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1)
            self.ax1.set_title("Actor Loss")
            self.ax2.set_title("Critic Loss")
            self.ax1.set_xlabel("Episodes")
            self.ax2.set_xlabel("Episodes")
            self.ax1.set_ylabel("Loss")
            self.ax2.set_ylabel("Loss")
            self.plot_created = True

        self.ax1.clear()
        self.ax2.clear()
        # Plot the losses where epochs are on the x-axis
        self.ax1.plot(self.losses_actor, label="Actor loss")
        self.ax1.plot(self.val_losses_actor, label="Actor val loss")
        self.ax2.plot(self.losses_critic, label="Critic loss")
        self.ax2.plot(self.val_losses_critic, label="Critic val loss")

        self.ax1.legend()
        self.ax2.legend()

        self.fig.savefig(f"images/losses_{self.board_size}x{self.board_size}.png")
