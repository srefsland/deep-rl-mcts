from itertools import combinations
from statemanager.hexstatemanager import HexStateManager
import matplotlib.pyplot as plt
from nn.boardgamenetcnn import BoardGameNetCNN
from .actor import Actor


def run_tournament(actors, num_games=25, board_size=4, temperature=1.0):
    combinations_pairs = list(combinations(actors, 2))
    agent_wins = {actor.name: 0 for actor in actors}

    for actor1, actor2 in combinations_pairs:
        actor1_wins = 0
        actor2_wins = 0

        for i in range(num_games):
            if i % 2 == 0:
                winner = run_game(
                    actor1, actor2, board_size=board_size, temperature=temperature)

                if winner == (1, 0):
                    actor1_wins += 1
                else:
                    actor2_wins += 1
            else:
                winner = run_game(
                    actor2, actor1, board_size=board_size, temperature=temperature)

                if winner == (1, 0):
                    actor2_wins += 1
                else:
                    actor1_wins += 1

        agent_wins[actor1.name] += actor1_wins
        agent_wins[actor2.name] += actor2_wins
        print(f"{actor1.name} vs {actor2.name}: {actor1_wins} - {actor2_wins}")

    # Display bar plot for each agent wins
    plt.bar(agent_wins.keys(), agent_wins.values())
    plt.show()


def run_game(actor1, actor2, board_size=4, temperature=1.0):
    board = HexStateManager(board_size=board_size)

    is_terminal = False
    while not is_terminal:
        current_player = board.player

        model_input = board.convert_to_nn_input()

        if current_player == (1, 0):
            move = actor1.predict_move(model_input, temperature=temperature)
        else:
            move = actor2.predict_move(model_input, temperature=temperature)

        board.make_move(move)

        is_terminal = board.check_winning_state()

    winner = current_player

    return winner


if __name__ == "__main__":
    board_size = 3
    model = BoardGameNetCNN(
        saved_model=f"models/model_{board_size}x{board_size}_0", board_size=board_size)
    model2 = BoardGameNetCNN(
        saved_model=f"models/model_{board_size}x{board_size}_50", board_size=board_size)
    model3 = BoardGameNetCNN(
        saved_model=f"models/model_{board_size}x{board_size}_100", board_size=board_size)
    model4 = BoardGameNetCNN(
        saved_model=f"models/model_{board_size}x{board_size}_150", board_size=board_size)
    model5 = BoardGameNetCNN(
        saved_model=f"models/model_{board_size}x{board_size}_200", board_size=board_size)

    agent1 = Actor("model_0", model, board_size=board_size)
    agent2 = Actor("model_50", model2, board_size=board_size)
    agent3 = Actor("model_100", model3, board_size=board_size)
    agent4 = Actor("model_150", model4, board_size=board_size)
    agent5 = Actor("model_200", model5, board_size=board_size)

    run_tournament([agent1, agent2, agent3, agent4, agent5],
                   board_size=board_size, temperature=0.5)
