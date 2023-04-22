import matplotlib.pyplot as plt
import numpy as np
import config

from actor import Actor
from display.hexboarddisplay import HexBoardDisplay
from nn.boardgamenetcnn import BoardGameNetCNN
from nn.boardgamenetann import BoardGameNetANN
from statemanager.hexstatemanager import HexStateManager

# NOTE: this is only for testing, not part of the actual delivery, disregard this file.


def visualize_one_game(actor1=None, actor2=None, board_size=4, best_move=False):
    board = HexStateManager(board_size=board_size)
    board_display = HexBoardDisplay()

    is_terminal = False
    while not is_terminal:
        current_player = board.player

        if current_player == (1, 0) and actor1 is not None:
            if best_move:
                move = actor1.predict_move(board, temperature=1.0)
            else:
                move = actor1.predict_move(board, temperature=0)

            new_move = board.make_move(move)
        elif current_player == (0, 1) and actor2 is not None:
            if best_move:
                move = actor2.predict_move(board, temperature=1.0)
            else:
                move = actor2.predict_move(board, temperature=0)

            new_move = board.make_move(move)
        else:
            new_move = board.make_random_move()

        board_display.display_board(board, delay=0.5, newest_move=new_move)

        is_terminal = board.check_winning_state(current_player)

    board_display.display_board(board, delay=0.5, winner=current_player)


def play_versus_actor(actor, board_size=4, best_move=True, player1=True):
    board = HexStateManager(board_size=board_size)
    board_display = HexBoardDisplay()

    new_move = None
    is_terminal = False
    while not is_terminal:
        board_display.display_board(board, delay=0.5, newest_move=new_move)
        current_player = board.player

        if current_player == (1, 0) and player1 or current_player == (0, 1) and not player1:
            x = input("Enter x: ")
            y = input("Enter y: ")

            move = (int(x), int(y))
        else:
            if best_move:
                move = actor.predict_move(board, temperature=1.0)
            else:
                move = actor.predict_move(board, temperature=0.0)

        new_move = board.make_move(move)
        is_terminal = board.check_winning_state(current_player)

    board_display.display_board(board, delay=0.5, winner=current_player)


def compare_models(actor1=None, actor2=None, board_size=4, best_move=False):
    wins_player_1 = 0
    n_games = 100

    for i in range(n_games):
        if i % 10 == 0:
            print(f"Game {i} / {n_games}")
        is_terminal = False
        board = HexStateManager(board_size=board_size)
        while not is_terminal:
            current_player = board.player

            if current_player == (1, 0) and actor1 is not None:
                if best_move:
                    move = actor1.predict_move(board, temperature=1.0)
                else:
                    move = actor1.predict_move(board, temperature=0)

                board.make_move(move)
            elif current_player == (0, 1) and actor2 is not None:
                if best_move:
                    move = actor2.predict_move(board, temperature=1.0)
                else:
                    move = actor2.predict_move(board, temperature=0)

                board.make_move(move)
            else:
                board.make_random_move()

            is_terminal = board.check_winning_state()

        wins_player_1 += 1 if current_player == (1, 0) else 0

    print(
        f"Player 1 won {wins_player_1} out of {n_games} games ({wins_player_1 / n_games * 100} %)")
    print(
        f"Player 2 won {n_games - wins_player_1} out of {n_games} games ({(n_games - wins_player_1) / n_games * 100} %)")


if __name__ == "__main__":
    actor1_episodes = 200
    actor2_episodes = 100

    saved_model1 = f"{config.MODEL_DIR}/model_{config.BOARD_SIZE}x{config.BOARD_SIZE}_{actor1_episodes}"
    saved_model2 = f"{config.MODEL_DIR}/model_{config.BOARD_SIZE}x{config.BOARD_SIZE}_{actor2_episodes}"
    model1 = (BoardGameNetANN(board_size=config.BOARD_SIZE, saved_model=saved_model1)
              if config.NN_TYPE == "ann"
              else BoardGameNetCNN(board_size=config.BOARD_SIZE, saved_model=saved_model1)
              )
    model2 = (BoardGameNetANN(board_size=config.BOARD_SIZE, saved_model=saved_model2)
              if config.NN_TYPE == "ann"
              else BoardGameNetCNN(board_size=config.BOARD_SIZE, saved_model=saved_model2)
              )
    actor1 = Actor("actor1", model1, board_size=config.BOARD_SIZE)
    actor2 = Actor("actor2", model2, board_size=config.BOARD_SIZE)

    mode = 'play'
    if mode == 'compare':
        compare_models(actor1=actor1, actor2=None,
                       board_size=config.BOARD_SIZE, best_move=True)
    elif mode == 'play':
        play_versus_actor(actor1, board_size=config.BOARD_SIZE,
                          best_move=True, player1=False)
    else:
        visualize_one_game(actor1=actor1, actor2=actor2,
                           board_size=config.BOARD_SIZE, best_move=True)
