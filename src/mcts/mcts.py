from . import MCTSNode
import numpy as np

class MCTS:
    def __init__(self, state_manager, c=1.0, max_iterations=1000):
        self.state_manager = state_manager
        self.c = c
        self.max_iterations = max_iterations
        self.root = MCTSNode(state_manager, None, None, None)