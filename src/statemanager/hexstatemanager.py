import copy

import numpy as np

from .hexboardcell import HexBoardCell
from .statemanager import StateManager

# Notes (for square board representation)
# Player 1 (red): top to bottom
# Player 2 (black): left to right


class HexStateManager(StateManager):
    def __init__(self, board_size=6, player=(1, 0), board=None):
        if board is None:
            self.board = self._initialize_state(board_size)
            self.board_size = board_size
        else:
            self.board = board
            self.board_size = len(board)

        # Whose turn it is
        self.player = player

    def _initialize_state(self, board_size):
        """Initializes state of the board.

        Args:
            board_size (int): size of the board.

        Returns:
            np.ndarray: the newly created board.
        """
        board = np.array([[HexBoardCell(row, col) for col in range(board_size)]
                         for row in range(board_size)])

        return board

    def copy_state(self):
        """Creates a deep copy of the current state of the game.

        Returns:
            _type_: new state manager with the same state as the current one.
        """
        return HexStateManager(player=self.player, board=copy.deepcopy(self.board))

    # NOTE: only passes player as parameter to be able to generalize for all types of state manager in 2v2 board games.
    # In Hex, the available moves are the same for both players.
    def get_moves_legal(self, player=None):
        """Fetches the legal moves for the current player, which is the empty cells.

        Args:
            player (tuple[int, int], optional): The player to get the moves for. Defaults to None.

        Returns:
            list[tuple[int, int]]: the current legal moves, represented as (x, y) coordinates.
        """
        moves = []

        for row in self.board:
            for node in row:
                if self.is_move_legal(node.position):
                    moves.append(node.position)

        return moves

    def print_board(self):
        """Prints the current state of the board to the terminal. Mostly for debugging purposes.
        """
        for row in self.board:
            for node in row:
                occupant = 1 if node.occupant == (1, 0) else 2 if node.occupant == (
                    0, 1) else 0
                print(occupant, end=' ')
            print()

        print()

    def is_move_legal(self, move):
        """Checks if the move is legal, i.e. if the cell is empty.

        Args:
            move (tuple[int, int]): the move to check.

        Returns:
            bool: true if the move is legal, false otherwise.
        """
        return self.board[move[0]][move[1]].is_empty()

    def make_random_move(self, player=None):
        """Makes a random move for the current player.

        Args:
            player (tuple[int, int], optional): the player to make the moves for. Defaults to None.

        Returns:
            tuple[int, int]: the randomly chosen move.
        """
        if player is None:
            player = self.player

        moves = self.get_moves_legal()

        if len(moves) == 0:
            return

        move = self.make_move(moves[np.random.randint(0, len(moves))], player)

        return move

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

        if self._is_within_bounds(move[0], move[1]) and self.is_move_legal(move):
            self.board[move[0]][move[1]].occupant = player
        else:
            raise Exception("Illegal move")

        self.player = (0, 1) if player == (1, 0) else (1, 0)

        return move

    # Inspired by the article here: https://www.idi.ntnu.no/emner/it3105/materials/neural/gao-2017.pdf
    # Should make it possible to feed to convolutional neural network with 5 channels, 3 for occupancy
    # and 2 for each player's turn
    def convert_to_nn_input(self):
        """Converts the game state to the input format of the convolutional neural network. 
        The 5 channels represent the game state as bitboards:
        Channel 1 is the cells occupied by player 1.
        Channel 2 is the cells occupied by player 2.
        Channel 3 is the cells currently unoccupied.
        Channel 4 are all 1's if the current player is 1.
        Channel 5 are all 1's if the current player is 2.

        Returns:
            np.ndarray: the nn input of shape (0, board_size, board_size, 5),
        """
        nn_input = np.zeros(
            shape=(self.board_size, self.board_size, 5), dtype=np.int8)

        is_occupied_player_1 = np.vectorize(
            lambda x: 1 if x.occupant == (1, 0) else 0)
        is_occupied_player_2 = np.vectorize(
            lambda x: 1 if x.occupant == (0, 1) else 0)
        is_occupied_empty = np.vectorize(
            lambda x: 1 if x.is_empty() else 0)

        nn_input[:, :, 0] = is_occupied_player_1(self.board)
        nn_input[:, :, 1] = is_occupied_player_2(self.board)
        nn_input[:, :, 2] = is_occupied_empty(self.board)
        nn_input[:, :, 3] = 1 if self.player == (1, 0) else 0
        nn_input[:, :, 4] = 1 if self.player == (0, 1) else 0

        nn_input = np.expand_dims(nn_input, axis=0)

        return nn_input

    def convert_to_diamond_shape(self):
        """Converts the game state to a diamond shape, which is the input format for displaying the board.
        In effect, this rotates the board 45 degrees clockwise.

        Example:
        [[1],
        [1, 2],
        [1, 2, 3],
        [1, 2],
        [1]]

        Returns:
            list: the converted board state.
        """
        diamond_array = []
        for i in range(-self.board_size + 1, self.board_size):
            diamond_array.append(np.diagonal(
                np.flipud(self.board), i).tolist())

        return diamond_array

    def _is_within_bounds(self, row, col):
        """Ensures that the current row and column are within the bounds of the board.

        Args:
            row (int): the row index.
            col (int): the column index.

        Returns:
            bool: true if within bounds, false if not.
        """
        return row >= 0 and row < self.board_size and col >= 0 and col < self.board_size

    def _expand_neighbors(self, node, player=None):
        """Finds neighbors that connect to the current node. Used to determine if the state is terminal (game over).

        Args:
            node (HexCell): the hexcell to expand neighbors to.
            player (tuple[int, int]), optional): _description_. Defaults to None.

        Returns:
            list[HexCell]: the neighbors that connect.
        """
        if player is None:
            player = self.player

        row, col = node.position

        neighbors_coords = [(row - 1, col), (row + 1, col), (row, col - 1),
                            (row, col + 1), (row + 1, col - 1), (row - 1, col + 1)]

        neighbors = []

        for neighbor in neighbors_coords:
            if self._is_within_bounds(neighbor[0], neighbor[1]) and self.board[neighbor[0]][neighbor[1]].occupant == player:
                neighbors.append(self.board[neighbor[0]][neighbor[1]])

        return neighbors

    def generate_child_states(self, player=None):
        """Generates all the possible child state of the current state of the game.

        Args:
            player (tuple[int, int]), optional): the player that is to move. Defaults to None.

        Returns:
            zip: the child states along with what move changed it.
        """
        if player is None:
            player = self.player

        child_states = []
        moves = self.get_moves_legal()

        for move in moves:
            child_board = self.copy_state()
            child_board.make_move(move, player)
            child_states.append(child_board)

        # Just makes it easier to keep track of what move was made to get to the child state
        return zip(child_states, moves)

    def check_winning_state(self, player=None):
        """Checks if there is a win in the current state of the board.

        Args:
            player (tuple[int, int], optional): the player to check for win. Defaults to None.

        Returns:
            bool: true if the player has won, false if not.
        """
        if player == (1, 0):
            return self._check_winning_state_player1()
        elif player == (0, 1):
            return self._check_winning_state_player2()
        else:
            return self._check_winning_state_player1() or self._check_winning_state_player2()

    # Player 1 (red) is top to bottom
    def _check_winning_state_player1(self):
        """Checks the winning state of player 1.

        Returns:
            bool: true if player 1 has won, false if not.
        """
        nodes_to_visit = []
        nodes_visited = []

        for col in self.board[0]:
            if col.occupant == (1, 0):
                nodes_to_visit.append(col)

        while len(nodes_to_visit) > 0:
            node = nodes_to_visit.pop(0)
            nodes_visited.append(node)

            if node.position[0] == self.board_size - 1:
                return True

            neighbors = self._expand_neighbors(node, (1, 0))

            for neighbor in neighbors:
                if neighbor not in nodes_to_visit and neighbor not in nodes_visited:
                    nodes_to_visit.append(neighbor)

        return False

    # Player 2 (black) is left to right
    def _check_winning_state_player2(self):
        """Checks the winning state of player 2.

        Returns:
            bool: true if player 2 has won, false if not.
        """
        nodes_to_visit = []
        nodes_visited = []

        for row in self.board:
            if row[0].occupant == (0, 1):
                nodes_to_visit.append(row[0])

        while len(nodes_to_visit) > 0:
            node = nodes_to_visit.pop(0)
            nodes_visited.append(node)

            if node.position[1] == self.board_size - 1:
                return True

            neighbors = self._expand_neighbors(node, (0, 1))

            for neighbor in neighbors:
                if neighbor not in nodes_to_visit and neighbor not in nodes_visited:
                    nodes_to_visit.append(neighbor)

        return False

    def has_winning_move(self, player=None):
        """Checks if any of the child states results in a win. Useful for 
        shortening the number of moves in each MCTS simulation.

        Args:
            player (tuple[int, int], optional): the player to check winning move for. Defaults to None.

        Returns:
            tuple[int, int]: the move that results in a win, None if there are none that results in a win.
        """
        if player is None:
            player = self.player

        moves = self.get_moves_legal()

        for move in moves:
            child_board = self.copy_state()
            child_board.make_move(move, player)

            if child_board.check_winning_state(player):
                return move

        return None

    def get_eval(self, winner=(1, 0)):
        """Passes the reward associated with a terminated game.

        Args:
            winner (tuple[int, int], optional): the winner of the game. Defaults to (1, 0).

        Returns:
            int: the reward that depends on which player is the winner.
        """
        return 1 if winner == (1, 0) else -1 if winner == (0, 1) else 0
