from display.hexboarddisplay import HexBoardDisplay
from statemanager.hexstatemanager import HexStateManager
from nn.boardgamenetcnn import BoardGameNetCNN
import numpy as np

if __name__ == "__main__":
    board = HexStateManager(3)
    board_display = HexBoardDisplay()

    model = BoardGameNetCNN(saved_model="models/model_3x3_100", board_size=3)
    model2 = BoardGameNetCNN(saved_model="models/model_3x3_50", board_size=3)

    is_terminal = False
    while not is_terminal:
        current_player = board.player

        if current_player == (0, 1):
            model_input = board.convert_to_nn_input()
            moves = model.predict(model_input)

            print(moves)

            move = np.argmax(moves)

            print(move)
            move = (move // 3, move % 3)

            new_move = board.make_move(move)
        else:
            model_input = board.convert_to_nn_input()
            moves = model2.predict(model_input)

            print(moves)

            move = np.argmax(moves)

            print(move)
            move = (move // 3, move % 3)

            new_move = board.make_move(move)

        board_display.display_board(
            board.convert_to_diamond_shape(), delay=0.5, newest_move=new_move)

        is_terminal = board.check_winning_state(current_player)

    board_display.display_board(
        board.convert_to_diamond_shape(), winner=current_player)
