import numpy as np
from collections import deque


class ReplayBuffer:
    def __init__(self, maxlen=400):
        # Deque should be more efficient than the previous list, since the time complexity of appending and popping from a deque is constant
        self.replay_buffer = deque(maxlen=maxlen)

    def clear(self):
        self.replay_buffer.clear()

    # A case should be a game state (root state of current game) combined with the target distribution D, derived from MCTS simulations
    def add_case(self, case):
        self.replay_buffer.append(case)

    def get_random_minibatch(self, batch_size):
        cases = self.replay_buffer

        if len(cases) > batch_size:
            row_idx = np.random.choice(len(cases), batch_size, replace=False)
            minibatch = [cases[i] for i in row_idx]
        else:
            minibatch = cases

        X = np.concatenate([x.astype(np.float32)
                           for x, _ in minibatch], axis=0)
        y = np.concatenate([y for _, y in minibatch], axis=0)

        return X, y
