from statemanager.hexboard import HexBoard
from statemanager.boarddisplay import BoardDisplay

if __name__ == "__main__":
    board = HexBoard(10)
    board_display = BoardDisplay()
    board.make_move((0, 0), (1, 0))
    board_display.visualize(board.convert_to_diamond_shape(), delay=0.5)
    board.make_move((1, 0), (0, 1))
    board_display.visualize(board.convert_to_diamond_shape(), delay=0.5)
    board.make_move((0, 1), (1, 0))
    board_display.visualize(board.convert_to_diamond_shape(), delay=0.5)
    board.make_move((2, 0), (0, 1))
    board_display.visualize(board.convert_to_diamond_shape(), delay=0.5)
    board.make_move((0, 2), (1, 0))
    board_display.visualize(board.convert_to_diamond_shape(), delay=0.5)

    board_display.visualize(board.convert_to_diamond_shape())
