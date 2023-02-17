from statemanager.hexstatemanager import HexStateManager
from display.hexboarddisplay import HexBoardDisplay
import copy

if __name__ == "__main__":
    board = HexStateManager(6)
    board_display = HexBoardDisplay()

    i = 0
    player = (1, 0)
    while board.check_winning_state(player) is False:
        player = (1, 0) if i % 2 == 0 else (0, 1)

        new_move = board.find_immediate_winning_move(player)

        if new_move is None:
            new_move = board.make_random_move(player)
        else:
            new_move = board.make_move(new_move, player)

        i += 1
        board_display.display_board(
            board.convert_to_diamond_shape(), delay=0.5, newest_move=new_move)

    board_display.display_board(
        board.convert_to_diamond_shape(), winner=player)
