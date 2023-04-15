from itertools import combinations
from statemanager.hexstatemanager import HexStateManager
import matplotlib.pyplot as plt
from nn.boardgamenetcnn import BoardGameNetCNN
from actor import Actor
from display.hexboarddisplay import HexBoardDisplay
import config


hexboard_display = HexBoardDisplay()


def run_tournament(actors, num_games=25, board_size=4, temperature=1.0):
    """Run the TOPP tournament for different actors.

    Args:
        actors (list[Actor]): the actors of different playing strengths.
        num_games (int, optional): number of games played between each actor. Defaults to 25.
        board_size (int, optional): the board size the actors are trained for. Defaults to 4.
        temperature (float, optional): the temperature means the likelihood of using the probability distribution
        versus the best move. Defaults to 1.0, which means the best move is taken always (highest percentage).
    """
    # Creates a combination such that each actor plays N games against all other actors.
    combinations_pairs = list(combinations(actors, 2))
    agent_wins = {actor.name: 0 for actor in actors}

    for actor1, actor2 in combinations_pairs:
        actor1_wins = 0
        actor2_wins = 0

        for i in range(num_games):
            display_game = i == num_games - 1
            if i % 2 == 0:
                # Display the last game of every series.
                winner = run_game(
                    actor1, actor2, board_size=board_size, temperature=temperature, display_game=display_game)

                if winner == (1, 0):
                    actor1_wins += 1
                else:
                    actor2_wins += 1
            else:
                # Display the last game of every series.
                winner = run_game(
                    actor2, actor1, board_size=board_size, temperature=temperature, display_game=display_game)

                if winner == (1, 0):
                    actor2_wins += 1
                else:
                    actor1_wins += 1

        agent_wins[actor1.name] += actor1_wins
        agent_wins[actor2.name] += actor2_wins
        if config.TOPP_VERBOSE:
            print(f"{actor1.name} vs {actor2.name}: {actor1_wins} - {actor2_wins}")

    total_games = sum(agent_wins.values())
    win_percentage = {agent: wins /
                      total_games for agent, wins in agent_wins.items()}
    plt.figure()
    # Display bar plot for each agent wins
    plt.title(f"TOPP Tournament {board_size}x{board_size} Win Percentages (%)")
    plt.bar(win_percentage.keys(), win_percentage.values())
    plt.show()


def run_game(actor1, actor2, board_size=4, temperature=1.0, display_game=False):
    """Run a game from the tournament.

    Args:
        actor1 (Actor): the first actor (player 1).
        actor2 (Actor): the second actor (player 2).
        board_size (int, optional): the board size the actors are trained on. Defaults to 4.
        temperature (float, optional): the temperature (best vs. probabilistic move). Defaults to 1.0.
        display_game (bool, optional): option to display the game. Defaults to False.
    Returns:
        tuple[int, int]: the winner of the game
    """
    board = HexStateManager(board_size=board_size)

    is_terminal = False
    while not is_terminal:
        current_player = board.player

        model_input = board.convert_to_nn_input()

        if current_player == (1, 0):
            move = actor1.predict_move(model_input, temperature=temperature)
        else:
            move = actor2.predict_move(model_input, temperature=temperature)

        move = board.make_move(move)

        is_terminal = board.check_winning_state()

        if display_game:
            winner = current_player if is_terminal else None
            hexboard_display.display_board(
                board.convert_to_diamond_shape(), delay=0.3, newest_move=move, winner=winner, actor1=actor1.name, actor2=actor2.name)

    winner = current_player

    return winner


if __name__ == "__main__":
    save_interval = config.NUM_EPISODES // (config.TOPP_M - 1)

    agents = []
    for i in range(config.TOPP_M):
        print(
            f"Loading model {i * save_interval} {config.BOARD_SIZE}x{config.BOARD_SIZE} from {config.MODEL_DIR}")
        model = BoardGameNetCNN(
            saved_model=f"{config.MODEL_DIR}/model_{config.BOARD_SIZE}x{config.BOARD_SIZE}_{i * save_interval}", board_size=config.BOARD_SIZE)
        agents.append(
            Actor(f"model_{i * save_interval}", model, board_size=config.BOARD_SIZE))

    run_tournament(agents, num_games=config.TOPP_NUM_GAMES,
                   board_size=config.BOARD_SIZE, temperature=config.TOPP_TEMPERATURE)
