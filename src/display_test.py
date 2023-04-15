from display.hexboarddisplay import HexBoardDisplay
from statemanager.hexstatemanager import HexStateManager
from nn.boardgamenetcnn import BoardGameNetCNN
import numpy as np

# NOTE: this is only for testing, not part of the actual delivery, disregard this file.


def visualize_one_game(actor1_episodes=100, actor2_episodes=50, board_size=4, random_player1=False, random_player2=False, best_move=False):
    board = HexStateManager(board_size=board_size)
    board_display = HexBoardDisplay()

    model = BoardGameNetCNN(
        saved_model=f"models/model_{board_size}x{board_size}_{actor1_episodes}", board_size=board_size)
    model2 = BoardGameNetCNN(
        saved_model=f"models/model_{board_size}x{board_size}_{actor2_episodes}", board_size=board_size)

    is_terminal = False
    while not is_terminal:
        current_player = board.player

        if current_player == (1, 0) and not random_player1:
            model_input = board.convert_to_nn_input()
            moves = model.predict(model_input).reshape(-1,)

            if best_move:
                move = np.argmax(moves)
                move = (move // board_size, move % board_size)
            else:
                indices = np.arange(len(moves))

                move = np.random.choice(indices, p=moves)
                move = (move // board_size, move % board_size)

            new_move = board.make_move(move)
        elif current_player == (0, 1) and not random_player2:
            model_input = board.convert_to_nn_input()
            moves = model2.predict(model_input).reshape(-1,)

            if best_move:
                move = np.argmax(moves)
                move = (move // board_size, move % board_size)
            else:
                indices = np.arange(len(moves))

                move = np.random.choice(indices, p=moves)
                move = (move // board_size, move % board_size)

            new_move = board.make_move(move)
        else:
            new_move = board.make_random_move()

        board_display.display_board(
            board.convert_to_diamond_shape(), delay=0.5, newest_move=new_move)

        is_terminal = board.check_winning_state(current_player)

    board_display.display_board(
        board.convert_to_diamond_shape(), winner=current_player)


def play_versus_actor(actor_episodes=100, board_size=4, best_move=True, player1=True):
    board = HexStateManager(board_size=board_size)
    board_display = HexBoardDisplay()

    model = BoardGameNetCNN(
        saved_model=f"models/model_{board_size}x{board_size}_{actor_episodes}", board_size=board_size)

    new_move = None
    is_terminal = False
    while not is_terminal:
        board_display.display_board(
            board.convert_to_diamond_shape(), delay=0.5, newest_move=new_move)
        current_player = board.player

        if current_player == (1, 0) and player1 or current_player == (0, 1) and not player1:
            x = input("Enter x: ")
            y = input("Enter y: ")

            move = (int(x), int(y))
        else:
            model_input = board.convert_to_nn_input()
            moves = model.predict(model_input).reshape(-1,)

            if best_move:
                move = np.argmax(moves)
                move = (move // board_size, move % board_size)
            else:
                indices = np.arange(len(moves))

                move = np.random.choice(indices, p=moves)
                move = (move // board_size, move % board_size)

        new_move = board.make_move(move)

        is_terminal = board.check_winning_state(current_player)

    board_display.display_board(
        board.convert_to_diamond_shape(), winner=current_player)


def compare_models(actor1_episodes=100, actor2_episodes=50, board_size=4, random_player1=False, random_player2=False, best_move=False):
    model = BoardGameNetCNN(
        saved_model=f"models/model_{board_size}x{board_size}_{actor1_episodes}", board_size=board_size)
    model2 = BoardGameNetCNN(
        saved_model=f"models/model_{board_size}x{board_size}_{actor2_episodes}", board_size=board_size)

    wins_player_1 = 0
    n_games = 100

    for i in range(n_games):
        if i % 10 == 0:
            print(f"Game {i} / {n_games}")
        is_terminal = False
        board = HexStateManager(board_size=board_size)
        while not is_terminal:
            current_player = board.player

            if current_player == (1, 0) and not random_player1:
                model_input = board.convert_to_nn_input()
                moves = model.call(model_input).reshape(-1,)

                if best_move:
                    move = np.argmax(moves)
                    move = (move // board_size, move % board_size)
                else:
                    indices = np.arange(len(moves))

                    move = np.random.choice(indices, p=moves)
                    move = (move // board_size, move % board_size)

                new_move = board.make_move(move)
            elif current_player == (0, 1) and not random_player2:
                model_input = board.convert_to_nn_input()
                moves = model2.call(model_input).reshape(-1,)

                if best_move:
                    move = np.argmax(moves)
                    move = (move // board_size, move % board_size)
                else:
                    indices = np.arange(len(moves))

                    move = np.random.choice(indices, p=moves)
                    move = (move // board_size, move % board_size)

                new_move = board.make_move(move)
            else:
                new_move = board.make_random_move()

            is_terminal = board.check_winning_state()

        wins_player_1 += 1 if current_player == (1, 0) else 0

    print(
        f"Player 1 won {wins_player_1} out of {n_games} games ({wins_player_1 / n_games * 100} %)")
    print(
        f"Player 2 won {n_games - wins_player_1} out of {n_games} games ({(n_games - wins_player_1) / n_games * 100} %)")


if __name__ == "__main__":
    mode = 'playd'
    if mode == 'compare':
        compare_models(60, 100, 5, random_player1=False,
                       random_player2=False, best_move=False)
    elif mode == 'play':
        play_versus_actor(200, 5, best_move=True, player1=True)
    else:
        visualize_one_game(160, 100, 7, random_player1=False,
                           random_player2=False, best_move=True)
