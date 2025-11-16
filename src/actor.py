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
        batch_worker=None,
    ):
        self.name = name
        self.nn = nn
        self.board_size = board_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_critic = epsilon_critic
        self.epsilon_decay_critic = epsilon_decay_critic
        self.batch_worker = batch_worker

    def epsilon_greedy_policy(self, state: np.ndarray, player: int, legal_moves: set):
        if np.random.random() < self.epsilon:
            move = self.predict_random_move(legal_moves)
        elif self.batch_worker is not None:
            move = self.get_move_from_batch_inference(state, player, legal_moves)
        else:
            move = self.predict_best_move(state, player, legal_moves)

        return move

    def get_move_from_batch_inference(self, state, player, legal_moves):
        nn_input = convert_board_state_to_tensor(state, player)
        result_queue = self.batch_worker.infer(nn_input)
        move_probs = result_queue.get()
        filtered = self.sanitize_move_probs(move_probs, legal_moves)

        return self.select_best_move_from_probs(filtered)

    def sanitize_move_probs(self, move_probs, legal_moves):
        # Only squeeze axis=0 if its size is 1, else leave as is
        if move_probs.shape[0] == 1:
            prediction = np.squeeze(move_probs, axis=0)
        else:
            prediction = move_probs
        for i in range(len(prediction)):
            mv = (i // self.board_size, i % self.board_size)
            if mv not in legal_moves:
                prediction[i] = 0
        return prediction

    def select_best_move_from_probs(self, prediction):
        move_idx = np.argmax(prediction)

        move = (move_idx // self.board_size, move_idx % self.board_size)

        return move

    def select_probabilistic_move_from_probs(self, prediction):
        probs = prediction.flatten()

        indices = np.arange(len(probs))

        probs = (
            probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs) / len(probs)
        )
        move_idx = np.random.choice(indices, p=probs)

        move = (move_idx // self.board_size, move_idx % self.board_size)

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
        predictions = self.nn.call_actor(nn_input)

        filtered = self.sanitize_move_probs(predictions, legal_moves)

        return self.select_best_move_from_probs(filtered)

    def predict_probabilistic_move(
        self, state: np.ndarray = None, player: int = None, legal_moves: set = None
    ):
        nn_input = convert_board_state_to_tensor(state, player)
        predictions = self.nn.call_actor(nn_input)

        filtered = self.sanitize_move_probs(predictions, legal_moves)

        return self.select_probabilistic_move_from_probs(filtered)
