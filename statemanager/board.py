import numpy as np
from hexnode import HexNode

class Board:
    def __init__(self, board_size):
        self.board = self.init_board(board_size)
        
    def init_board(self, board_size):
        board = np.array([[HexNode(x, y) for x in range(board_size)] for y in range(board_size)])
        
        return board
    
    def display_board(self):
        for row in self.board:
            for node in row:
                print(node.get_owner(), end=" ")
            print()