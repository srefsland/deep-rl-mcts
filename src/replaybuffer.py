import random
from collections import deque


class ReplayBufffer:
    def __init__(self, maxlen=400):
        # Deque should be more efficient than the previous list, since the time complexity of appending and popping from a deque is constant
        self.replay_buffer = deque(maxlen=maxlen)

    def clear(self):
        self.replay_buffer.clear()

    # A case should be a game state (root state of current game) combined with the target distribution D, derived from MCTS simulations
    def add_case(self, case):
        self.replay_buffer.append(case)

    def get_random_minibatch(self, batch_size):
        return random.sample(self.replay_buffer, batch_size) if len(self.replay_buffer) > batch_size else list(self.replay_buffer)
