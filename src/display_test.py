from display.hexboarddisplay import HexBoardDisplay
from statemanager.hexstatemanager import HexStateManager

if __name__ == "__main__":
    board = HexStateManager(6)
    board_display = HexBoardDisplay()

    is_terminal = False
    while not is_terminal:
        current_player = board.player

        new_move = board.make_random_move()
        board_display.display_board(
            board.convert_to_diamond_shape(), delay=0.5, newest_move=new_move)

        is_terminal = board.check_winning_state(current_player)

    board_display.display_board(
        board.convert_to_diamond_shape(), winner=current_player)
