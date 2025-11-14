import numpy as np

from nn.board_encoder import convert_board_state_to_tensor
from nn.hexresnet import HexResNet


class Actor:
    def __init__(
        self,
        name: str,
        nn: HexResNet,
        board_size: int,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.99,
        epsilon_critic: float = 2.0,
        epsilon_decay_critic: float = 0.996,
    ):
        self.name = name
        self.nn = nn
        self.board_size = board_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_critic = epsilon_critic
        self.epsilon_decay_critic = epsilon_decay_critic

    def epsilon_greedy_policy(self, state: np.ndarray, player: int, legal_moves: set):
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

    def predict_critic(self, state: np.ndarray, player_to_move: int):
        nn_input = convert_board_state_to_tensor(state, player_to_move)
        X = self.nn.call_critic(nn_input)

        return X.item()

    def predict_random_move(self, legal_moves: set):
        legal_moves_list = list(legal_moves)
        move = legal_moves_list[np.random.choice(len(legal_moves_list))]

        return move

    def predict_best_move(
        self, state: np.ndarray = None, player: int = None, legal_moves: set = None
    ):
        nn_input = convert_board_state_to_tensor(state, player)
        predictions = self._predict_moves(nn_input, legal_moves)

        prediction = np.argmax(predictions)
        move = (prediction // self.board_size, prediction % self.board_size)

        return move

    def predict_probabilistic_move(
        self, state: np.ndarray = None, player: int = None, legal_moves: set = None
    ):
        nn_input = convert_board_state_to_tensor(state, player)

        moves = self._predict_moves(nn_input, legal_moves).flatten()
        indices = np.arange(len(moves))

        move = np.random.choice(indices, p=moves)
        move = (move // self.board_size, move % self.board_size)

        return move

    def _predict_moves(self, X: np.ndarray, legal_moves: set):
        # Convert to tensor
        prediction = self.nn.call_actor(X)

        prediction = np.squeeze(prediction, axis=0)

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
