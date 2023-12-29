import numpy as np
from nn.litemodel import LiteModel
import time

class Actor:
    """The actor class is used to make moves from trained neural networks."""

    def __init__(self,
                 name,
                 nn,
                 board_size,
                 epsilon=1.0,
                 epsilon_decay=0.99,
                 epsilon_critic=2.0,
                 epsilon_decay_critic=0.996,
                 litemodel=None):
        self.name = name
        self.nn = nn
        self.board_size = board_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_critic = epsilon_critic
        self.epsilon_decay_critic = epsilon_decay_critic
        self.litemodel = litemodel
        
        
    def epsilon_greedy_policy(self, state, player, legal_moves):
        if np.random.random() < self.epsilon:
            move = self.predict_random_move(legal_moves)
        else:
            move = self.predict_best_move(state, player, legal_moves)
        
        return move
            
    def decrease_epsilon(self):
        epsilon_decayed = self.epsilon * self.epsilon_decay
        epsilon_critic_decayed = self.epsilon_critic * self.epsilon_decay_critic
        
        self.epsilon = max(epsilon_decayed, 0.1)
        self.epsilon_critic = max(epsilon_critic_decayed, 0.1)
    
    def predict_critic(self, state, player):
        X = self.nn.convert_to_nn_input(state, player)
        prediction = self.nn.call_critic(X)
        
        return prediction
    
    def predict_random_move(self, legal_moves):
        # Get a random move from the legal moves set
        legal_moves_list = list(legal_moves)
        move = legal_moves_list[np.random.choice(len(legal_moves_list))]
        
        return move
        
    def predict_best_move(self, state=None, player=None, legal_moves=None):
        """Predicts the best move given the model input.

        Args:
            state (StateManager, optional): the state of the game. Defaults to None.
            model_input (np.ndarray, optional): the raw model input. Defaults to None.

        Returns:
            tuple[int, int]: the move to choose
        """
        nn_input = self.nn.convert_to_nn_input(state, player)
        predictions = self._predict_moves(nn_input, legal_moves)

        prediction = np.argmax(predictions)
        move = (prediction // self.board_size, prediction % self.board_size)

        return move

    def predict_probabilistic_move(self, state=None, player=None, legal_moves=None):
        """Predicts the move according to the probability distribution given by the model.

        Args:
            state (StateManager, optional): the state of the game. Defaults to None.
            model_input (np.ndarray, optional): the raw model input. Defaults to None.

        Returns:
            tuple[int, int]: the move to choose
        """
        nn_input = self.nn.convert_to_nn_input(state, player)

        moves = self._predict_moves(nn_input, legal_moves).flatten()
        indices = np.arange(len(moves))

        move = np.random.choice(indices, p=moves)
        move = (move // self.board_size, move % self.board_size)

        return move
    
    def _predict_moves(self, X, legal_moves):
        """Predicts the output of the neural network given the input.
        Uses the __call__ method of the model, which is faster than using the predict method.

        Args:
            X (np.ndarray): the input to the neural network

        Returns:
            np.ndarray: the predictions for each cell
        """
        # Convert to tensor
        if self.litemodel is None:
            prediction = self.nn.call_actor(X)
        else:
            prediction = self.litemodel.predict_single(np.squeeze(X, axis=0))

        for i in range(len(prediction)):
            move = (i // self.board_size, i % self.board_size)
            if move not in legal_moves:
                prediction[i] = 0

        sum_prediction = np.sum(prediction)
        # If the sum of the prediction is zero, then the mask is used as a fallback
        # to still return a valid move. This can happen when the model predicts something
        # to be zero.
        predictions_normalized = prediction / max(sum_prediction, 1e-6)
        return predictions_normalized.reshape((self.board_size, self.board_size))
    
    def create_lite_model(self):
        self.litemodel = LiteModel.from_keras_model(self.nn.model)
        
    def update_lite_model(self, litemodel):
        self.litemodel = litemodel
        
    def train_model(self, X, y_actor, y_critic, epochs=10, batch_size=32):
        self.nn.fit(X, y_actor, y_critic, epochs=epochs, batch_size=batch_size)
