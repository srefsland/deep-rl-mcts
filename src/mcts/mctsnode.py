class MCTSNode:
    def __init__(self, state, player_to_move, move=None, parent=None):
        self.parent = parent
        self.state = state
        self.player_to_move = player_to_move
        self.children = None
        # Accumulated rewards
        self.e = 0
        self.n = 0
        # The move that led to this node
        self.move = move

    def update_values(self, reward):
        self.n += 1
        self.e += reward

    def is_leaf_node(self):
        return self.children is None

    def is_root(self):
        return self.parent is None
    
    def get_qsa(self):
        # Normalize accumulated reward by number of visits
        return self.e / self.n if self.n > 0 else 0
