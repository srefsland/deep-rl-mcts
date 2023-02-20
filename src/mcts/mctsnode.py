class MTCSNode:
    def __init__(self, state, player, parent=None):
        self.parent = parent
        self.state = state
        self.player = player
        self.children = []
        self.qsa = {}
        self.nsa = {}
        self.n = 0

    def expand(self):
        for child_state in self.state.generate_child_states(self.player):
            self.children.append(
                MTCSNode(child_state, self.player, self))
            self.QSA[child_state] = 0
            self.NSA[child_state] = 0
