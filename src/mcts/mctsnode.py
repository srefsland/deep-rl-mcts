import threading
from typing import Optional, Tuple


class MCTSNode:
    def __init__(
        self,
        move: Optional[Tuple[int, int]] = None,
        parent: Optional["MCTSNode"] = None,
        player_to_move: Optional[int] = None,
        use_locks: bool = False,
    ):
        self.parent = parent
        self.children = None
        self.children_lock = threading.Lock() if use_locks else None
        self.backprop_lock = threading.Lock() if use_locks else None
        # Accumulated rewards
        self.e = 0.0
        self.n = 0
        # The move that led to this node
        self.move = move
        self.player_to_move = player_to_move

    def update_values(self, reward: float):
        self.n += 1
        self.e += reward

    def is_leaf_node(self):
        return self.children is None

    def is_root(self):
        return self.parent is None

    def get_qsa(self):
        # Normalize accumulated reward by number of visits
        return self.e / self.n if self.n > 0 else 0
