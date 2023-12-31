import pickle

import numpy as np
from disjoint_set import DisjointSet

from .statemanager import StateManager


class HexStateManager(StateManager):
    def __init__(self, board_size=6, **kwargs):
        self.switch_rule_allowed = kwargs.get("switch_rule_allowed", True)
        self.board_size = board_size
        self._initialize_state(board_size)

    def copy_state_manager(self):
        """Creates a deep copy of the current state of the game.

        Returns:
            HexStateManager: new state manager with the same state as the current one.
        """
        return pickle.loads(pickle.dumps(self))
        
    # NOTE: only passes player as parameter to be able to generalize for all types of state manager in 2v2 board games.
    # In Hex, the available moves are the same for both players.
    def get_legal_moves(self, player=None):
        """Fetches the legal moves for the current player, which are the empty cells.

        Args:
            player (int, optional): The player to get the moves for. Defaults to None.

        Returns:
            list[tuple[int, int]]: the current legal moves, represented as (x, y) coordinates.
        """
        return self.legal_moves

    def make_move(self, move, player=None):
        """Update the game state by making the provided move.

        Args:
            move (tuple[int, int]): the move to be made.
            player (tuple[int, int], optional): the player that makes the move. Defaults to None.

        Raises:
            Exception: is raised if move is not legal (i.e. a non empty cell).

        Returns:
            tuple[int, int]: the move that was made.
        """
        if player is None:
            player = self.player

        if move not in self.legal_moves:
            raise Exception("Illegal move")
        
        if len(self.move_history) > 1:
            self.legal_moves.remove(move)
        elif len(self.move_history) == 1:
            if move in self.moves_made:
                self.legal_moves.remove(move)
                self.switched = True
            else:
                self.legal_moves -= self.moves_made
                self.legal_moves.remove(move)
        elif len(self.move_history) == 0 and not self.switch_rule_allowed:
            self.legal_moves.remove(move)
        
        self.moves_made.update([move])
        self.move_history.append((move, player))

        if not (len(self.move_history) == 2 and self.switched):
            self.board[move[0]][move[1]] = player
            
            neighbors = self._expand_neighbors(move, player)
            for neighbor in neighbors:
                if player == 1:
                    self.disjoint_set_red.union(neighbor, move)
                else:
                    self.disjoint_set_blue.union(neighbor, move)
        
            self.player = -player

        return move

    def make_random_move(self, player=None):
        """Makes a random move for the current player.

        Args:
            player (int, optional): the player to make the moves for. Defaults to None.

        Returns:
            tuple[int, int]: the randomly chosen move.
        """
        if player is None:
            player = self.player

        moves = self.get_legal_moves()

        if len(moves) == 0:
            return

        move = self.make_move(list(moves)[np.random.randint(0, len(moves))], player)

        return move

    def generate_child_states(self, player=None):
        """Generates all the child states of the current state.

        Args:
            player (int, optional): the player of the current state. Defaults to None.

        Yields:
            tuple: child board, child player and move that was made to get to the child board.
        """
        if player is None:
            player = self.player

        moves = self.get_legal_moves()

        for move in moves:
            state_manager = self.copy_state_manager()
            state_manager.make_move(move, player)
            
            node_player = state_manager.player if not state_manager.switched else -state_manager.player
            
            yield state_manager.board, node_player, move

    def check_winning_state(self, player=None):
        """Checks if there is a win in the current state of the board.

        Args:
            player (int, optional): the player to check for win. Defaults to None.

        Returns:
            bool: true if the player has won, false if not.
        """
        if player == 1:
            return self._check_winning_state_player1()
        elif player == -1:
            return self._check_winning_state_player2()
        else:
            return (
                self._check_winning_state_player1()
                or self._check_winning_state_player2()
            )

    def reset(self):
        self._initialize_state(self.board_size)

    def get_eval(self, winner=1):
        """Passes the reward associated with a terminated game.

        Args:
            winner (tuple[int, int], optional): the winner of the game. Defaults to (1, 0).

        Returns:
            int: the reward that depends on which player is the winner.
        """
        return winner if not self.switched else -winner
    
    def get_distribution_shape(self):
        return np.zeros((self.board_size, self.board_size))

    def print_board(self):
        """Prints the current state of the board to the terminal. Mostly for debugging purposes."""
        for row in self.board:
            for cell in row:
                occupant = 1 if cell == 1 else 2 if cell == -1 else 0
                print(occupant, end=" ")
            print()

        print()

    def _initialize_state(self, board_size):
        """Initializes state of the board.

        Args:
            board_size (int): size of the board.

        Returns:
            np.ndarray: the newly created board.
        """
        self.board = np.zeros((board_size, board_size))
        self.switched = False
        self.legal_moves = set([(i, j) for j in range(board_size) for i in range(board_size)])
        self.moves_made = set()
        self.move_history = []
        self.player = 1
        
        self.top_node = (-1, 0)
        self.bottom_node = (board_size, 0)
        self.left_node = (0, -1)
        self.right_node = (0, board_size)
        
        cells = [(i, j) for j in range(board_size) for i in range(board_size)]
        self.disjoint_set_red = DisjointSet(cells + [self.top_node, self.bottom_node])
        self.disjoint_set_blue = DisjointSet(cells + [self.left_node, self.right_node])
        
        for i in range(board_size):
            self.disjoint_set_red.union((0, i), self.top_node)
            self.disjoint_set_red.union((board_size-1, i), self.bottom_node)
            self.disjoint_set_blue.union((i, 0), self.left_node)
            self.disjoint_set_blue.union((i, board_size-1), self.right_node)

    def _check_winning_state_player1(self):
        """Checks the winning state of player 1.

        Returns:
            bool: true if player 1 has won, false if not.
        """
        return self.disjoint_set_red.find(self.top_node) == self.disjoint_set_red.find(self.bottom_node)

    def _check_winning_state_player2(self):
        """Checks the winning state of player 2.

        Returns:
            bool: true if player 2 has won, false if not.
        """
        return self.disjoint_set_blue.find(self.left_node) == self.disjoint_set_blue.find(self.right_node)

    def _expand_neighbors(self, cell, player=None):
        """Finds neighbors that connect to the current node. Used to determine if the state is terminal (game over).

        Args:
            cell (tuple[int, int]): the hexcell to expand neighbors to.
            player (int), optional): _description_. Defaults to None.

        Returns:
            list[tuple[int, int]]: the neighbors that connect.
        """
        if player is None:
            player = self.player

        row, col = cell

        neighbors_coords = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
            (row + 1, col - 1),
            (row - 1, col + 1),
        ]

        neighbors = []
        # Select everyone but index 1
        move_history = self.move_history if not self.switched else self.move_history[0:1] + self.move_history[2:]

        for neighbor in neighbors_coords:
            if (neighbor, player) in move_history:
                neighbors.append(neighbor)

        return neighbors
