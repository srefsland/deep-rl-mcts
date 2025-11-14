from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayCase:
    board_state: np.ndarray
    target_distribution: np.ndarray
    target_value: np.ndarray


@dataclass
class Batch:
    X: np.ndarray
    y_actor: np.ndarray
    y_critic: np.ndarray


class ReplayBuffer:
    def __init__(self, maxlen: int = 700):
        self.replay_buffer = deque(maxlen=maxlen)

    def is_full(self):
        return len(self.replay_buffer) == self.replay_buffer.maxlen

    def clear(self):
        self.replay_buffer.clear()

    # A case should be a game state (root state of current game) combined with the target distribution D, derived from MCTS simulations
    def add_case(self, case: ReplayCase):
        self.replay_buffer.append(case)

    def get_random_minibatch(self, batch_size: int) -> Batch:
        cases = list(self.replay_buffer)
        if batch_size >= len(cases):
            minibatch = cases
        else:
            row_idx = np.random.choice(len(cases), size=batch_size, replace=False)
            minibatch = [cases[i] for i in row_idx]

        X = np.concatenate(
            [case.board_state.astype(np.float32) for case in minibatch], axis=0
        )
        y_actor = np.concatenate(
            [case.target_distribution for case in minibatch], axis=0
        )
        y_critic = np.concatenate([case.target_value for case in minibatch], axis=0)
        if y_critic.ndim == 1:
            y_critic = np.expand_dims(y_critic, axis=1)

        return Batch(X=X, y_actor=y_actor, y_critic=y_critic)

    def get_all_cases(self) -> Batch:
        cases = list(self.replay_buffer)

        X = np.concatenate(
            [case.board_state.astype(np.float32) for case in cases], axis=0
        )
        y_actor = np.concatenate([case.target_distribution for case in cases], axis=0)
        y_critic = np.concatenate([case.target_value for case in cases], axis=0)
        if y_critic.ndim == 1:
            y_critic = np.expand_dims(y_critic, axis=1)

        return Batch(X=X, y_actor=y_actor, y_critic=y_critic)
