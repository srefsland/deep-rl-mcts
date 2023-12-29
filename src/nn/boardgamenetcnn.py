import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from . import nn_options


class BoardGameNetCNN:
    def __init__(
        self,
        convolutional_layers=(32, 64),
        lr=0.001,
        activation="relu",
        output_activation_actor="softmax",
        output_activation_critic="tanh",
        loss_actor="categorical_crossentropy",
        loss_critic="mse",
        optimizer="Adam",
        board_size=6,
        bridge_features=False,
        saved_model=None,
    ):
        self.convolutional_layers = convolutional_layers
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
        self.bridge_features = bridge_features

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
        input_layer = tf.keras.layers.Input(shape=(self.board_size, self.board_size, 7 if self.bridge_features else 5))
        hidden_layer = input_layer
        
        hidden_layer = tf.keras.layers.Conv2D(
            self.convolutional_layers[0], (5, 5), padding="same",
            kernel_regularizer=tf.keras.regularizers.l2(0.001)
        )(hidden_layer)
        hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)
        hidden_layer = self.activation(hidden_layer)
        
        for i in range(1, len(self.convolutional_layers)):
            hidden_layer = tf.keras.layers.Conv2D(
                self.convolutional_layers[i], (3, 3), padding="same",
                kernel_regularizer=tf.keras.regularizers.l2(0.001)
            )(hidden_layer)
            hidden_layer = tf.keras.layers.BatchNormalization()(hidden_layer)
            hidden_layer = self.activation(hidden_layer)
        
        output_actor = tf.keras.layers.Conv2D(1, kernel_size=(1, 1))(hidden_layer)
        output_actor = tf.keras.layers.BatchNormalization()(output_actor)
        output_actor = self.activation(output_actor)
        output_actor = tf.keras.layers.Flatten()(output_actor)
        output_actor = tf.keras.layers.Activation(self.output_activation_actor, name="actor")(output_actor)
        
        output_critic = tf.keras.layers.Conv2D(1, kernel_size=(1, 1))(hidden_layer)
        output_critic = tf.keras.layers.BatchNormalization()(output_critic)
        output_critic = self.activation(output_critic)
        output_critic = tf.keras.layers.Flatten()(output_critic)
        output_critic = tf.keras.layers.Dense(units=64, activation=self.activation)(output_critic)
        output_critic = tf.keras.layers.Dense(units=1, activation=self.output_activation_critic, name="critic")(output_critic)

        model = tf.keras.models.Model(
            inputs=input_layer, outputs=[output_actor, output_critic]
        )
        model.compile(
            optimizer=self.optimizer, loss=[self.loss_actor, self.loss_critic]
        )
        self.model = model
        self.model.summary()

    def fit(self, X, y_actor, y_critic, epochs=10, batch_size=32):
        """Fits the model.

        Args:
            X (np.ndarray): the training data (game states).
            y_actor (np.ndarray): the target distribution for the actor network.
            y_critic (np.ndarray): the target distribution for the critic network.
            epochs (int, optional): number of epochs in training. Defaults to 10.
            batch_size (int, optional): the batch size. Defaults to 32.
        """
        early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, min_delta=0.001)

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

    def call_actor(self, X):
        """Predicts the output of the neural network given the input.
        Uses the __call__ method of the model, which is faster than using the predict method.

        Args:
            X (np.ndarray): the input to the neural network

        Returns:
            np.ndarray: the predictions for each cell
        """
        # Convert to tensor
        X = tf.convert_to_tensor(X)
        prediction = self.model(X)[0]

        # Convert output to numpy array
        prediction = prediction.numpy()

        return prediction

    def call_critic(self, X):
        """Predicts the output of the neural network given the input.
        Uses the __call__ method of the model, which is faster than using the predict method.

        Args:
            X (np.ndarray): the input to the neural network

        Returns:
            np.ndarray: the predictions for each cell
        """
        # Convert to tensor
        X = tf.convert_to_tensor(X)
        prediction = self.model(X)[1]

        # Convert output to numpy array
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
        Channel 

        Returns:
            np.ndarray: the nn input of shape (0, board_size, board_size, 5),
        """
        nn_input = np.zeros(
            shape=(self.board_size, self.board_size, 7 if self.bridge_features else 5), dtype=np.int8)

        is_occupied_player_1 = np.vectorize(lambda x: 1 if x == 1 else 0)
        is_occupied_player_2 = np.vectorize(lambda x: 1 if x == -1 else 0)
        is_occupied_empty = np.vectorize(lambda x: 1 if x == 0 else 0)

        nn_input[:, :, 0] = is_occupied_player_1(state)
        nn_input[:, :, 1] = is_occupied_player_2(state)
        nn_input[:, :, 2] = is_occupied_empty(state)
        nn_input[:, :, 3] = 1 if player == 1 else 0
        nn_input[:, :, 4] = 1 if player == -1 else 0
        
        if self.bridge_features:
            nn_input[:, :, 5] = 0
            nn_input[:, :, 6] = 0
            
            bridge_pattern1 = np.array([[1, 0], [0, 1]])
            bridge_pattern2 = np.array([[1, 0], [0, 1]])
            bridge_pattern3 = np.array([[0, 1], [1, 0]])
            
            for i in range(self.board_size-1):
                for j in range(self.board_size-1):
                    bridge_pattern_p1 = nn_input[i:i+2, j:j+2, 0]
                    bridge_pattern_p2 = nn_input[i:i+2, j:j+2, 1]
                    
                    if np.all((bridge_pattern_p1 & bridge_pattern1) | bridge_pattern_p1 == bridge_pattern1):
                        nn_input[i:i+2, j:j+2, 5] = ~bridge_pattern1.astype(bool)
                    
                    if np.all((bridge_pattern_p2 & bridge_pattern1) | bridge_pattern_p2 == bridge_pattern1):
                        nn_input[i:i+2, j:j+2, 6] = ~bridge_pattern1.astype(bool)
                    
            for i in range(self.board_size-2):
                for j in range(1, self.board_size):
                    bridge_pattern_p1 = np.array([[nn_input[i, j, 0], nn_input[i+1, j-1, 0]],
                                                [nn_input[i+1, j, 0],nn_input[i+2, j-1, 0]]])
                    bridge_pattern_p2 = np.array(([nn_input[i, j, 1], nn_input[i+1, j-1, 1]],
                                                [nn_input[i+1, j, 1],nn_input[i+2, j-1, 1]]))
                    
                    if np.all((bridge_pattern_p1 & bridge_pattern2) | bridge_pattern_p1 == bridge_pattern2):
                        nn_input[i+1, j-1, 5] = 1
                        nn_input[i+1, j, 5] = 1
                        
                    if np.all((bridge_pattern_p2 & bridge_pattern2) | bridge_pattern_p2 == bridge_pattern2):
                        nn_input[i+1, j-1, 6] = 1
                        nn_input[i+1, j, 6] = 1
            
            for i in range(1, self.board_size):
                for j in range(self.board_size-2):
                    bridge_pattern_p1 = np.array([[nn_input[i-1, j+1, 0], nn_input[i-1, j+2, 0]],
                                                [nn_input[i, j, 0],nn_input[i, j+1, 0]]])
                    bridge_pattern_p2 = np.array([[nn_input[i-1, j+1, 1], nn_input[i-1, j+2, 1]],
                                                [nn_input[i, j, 1],nn_input[i, j+1, 1]]])
                    
                    if np.all((bridge_pattern_p1 & bridge_pattern3) | bridge_pattern_p1 == bridge_pattern3):
                        nn_input[i-1, j+1, 5] = 1
                        nn_input[i, j+1, 5] = 1
                        
                    if np.all((bridge_pattern_p2 & bridge_pattern3) | bridge_pattern_p2 == bridge_pattern3):
                        nn_input[i-1, j+1, 6] = 1
                        nn_input[i, j+1, 6] = 1

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
