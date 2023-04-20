from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, maxlen=700):
        # Deque should be more efficient than the previous list, since the time complexity of appending and popping from a deque is constant
        self.replay_buffer = deque(maxlen=maxlen)

    def clear(self):
        """Clears the replay buffer, removing all the cases.
        """
        self.replay_buffer.clear()

    # A case should be a game state (root state of current game) combined with the target distribution D, derived from MCTS simulations
    def add_case(self, case):
        """Adds a case, which consists of a root state and a distribution for all moves.

        Args:
            case (tuple[np.ndarray, np.ndarray]): the root state and distribution.
        """
        self.replay_buffer.append(case)

    def get_random_minibatch(self, batch_size):
        """Fetches a random minibatch from the replay buffer with weighted probability.

        Args:
            batch_size (int): the size to sample from.

        Returns:
            tuple[int, int]: the training samples along with the target distributions.
        """
        cases = list(self.replay_buffer)
        if batch_size >= len(cases):
            minibatch = cases
        else:
            # Sample cases using weighted probability, idea is to sample more recent cases often
            row_idx = np.random.choice(len(cases), size=batch_size, replace=False)
            minibatch = [cases[i] for i in row_idx]

        X = np.concatenate([x.astype(np.float32)
                        for x, _ in minibatch], axis=0)
        y = np.concatenate([y for _, y in minibatch], axis=0)

        return X, y
