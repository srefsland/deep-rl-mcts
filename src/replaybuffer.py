import random


class ReplayBufffer:
    def __init__(self, maxlen=400):
        self.maxlen = maxlen
        self.replay_buffer = []

    def clear(self):
        self.replay_buffer = []

    # A case should be a game state (root state of current game) combined with the target distribution D, derived from MCTS simulations
    def add_case(self, case):
        self.replay_buffer.append(case)

        if len(self.replay_buffer) > self.maxlen:
            self.replay_buffer.pop(0)

    def get_random_minibatch(self, batch_size):
        return random.sample(self.replay_buffer, batch_size) if len(self.replay_buffer) > batch_size else self.replay_buffer
