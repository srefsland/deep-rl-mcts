import numpy as np
from boardcell import BoardCell

# Notes (for square board representation)
# Player 1 (red): top to bottom
# Player 2 (black): left to right


class HexBoard:
    def __init__(self, board_size=0, board=None):
        if board is None:
            self.board = self.init_board(board_size)
            self.board_size = board_size
        else:
            self.board = board
            self.board_size = len(board)

    def init_board(self, board_size):
        board = np.array([[BoardCell(row, col) for col in range(board_size)]
                         for row in range(board_size)])

        return board

    def display_board(self):
        for row in self.board:
            for node in row:
                print(node.get_owner(), end=" ")
            print()

    def get_moves_legal(self):
        moves = []

        for row in self.board:
            for node in row:
                if node.get_owner() == (0, 0):
                    moves.append(node.get_position())

        return moves

    def is_move_legal(self, move):
        if self.board[move[0]][move[1]].get_owner() == (0, 0):
            return True
        else:
            return False

    def make_move(self, move, player):
        if self.is_move_legal(move):
            self.board[move[0]][move[1]].set_owner(player)
        else:
            raise Exception("Illegal move")

    def convert_to_1D_array(self):
        return self.board.flatten()

    def convert_to_diamond_shape(self):
        diamond_array = []
        for i in range(-self.board_size + 1, self.board_size):
            diamond_array.append(np.diagonal(
                np.flipud(self.board), i).tolist())

        return diamond_array

    def expand_neighbors(self, node, player):
        row, col = node.get_position()

        neighbors_coords = [(row - 1, col), (row + 1, col), (row, col - 1),
                            (row, col + 1), (row + 1, col - 1), (row - 1, col + 1)]

        neighbors = []

        for neighbor in neighbors_coords:
            if neighbor[0] >= 0 and neighbor[0] < self.board_size and neighbor[1] >= 0 and neighbor[1] < self.board_size:
                if (self.board[neighbor[0]][neighbor[1]].get_owner() == player):
                    neighbors.append(self.board[neighbor[0]][neighbor[1]])

        return neighbors

    # Player 1 (red) is top to bottom
    def check_winning_state_player1(self):
        nodes_to_visit = []
        nodes_visited = []

        for col in self.board[0]:
            if col.get_owner() == (1, 0):
                nodes_to_visit.append(col)

        while len(nodes_to_visit) > 0:
            node = nodes_to_visit.pop(0)
            nodes_visited.append(node)

            if node.get_position()[0] == self.board_size - 1:
                return True

            neighbors = self.expand_neighbors(node, (1, 0))

            for neighbor in neighbors:
                if neighbor not in nodes_to_visit and neighbor not in nodes_visited:
                    nodes_to_visit.append(neighbor)

        return False

    # Player 2 (black) is left to right
    def check_winning_state_player2(self):
        nodes_to_visit = []
        nodes_visited = []

        for row in self.board:
            if row[0].get_owner() == (0, 1):
                nodes_to_visit.append(row[0])

        while len(nodes_to_visit) > 0:
            node = nodes_to_visit.pop(0)
            nodes_visited.append(node)

            if node.get_position()[1] == self.board_size - 1:
                return True

            neighbors = self.expand_neighbors(node, (0, 1))

            for neighbor in neighbors:
                if neighbor not in nodes_to_visit and neighbor not in nodes_visited:
                    nodes_to_visit.append(neighbor)

        return False