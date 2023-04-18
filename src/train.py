import logging
import time

from tqdm import tqdm

import config
import replay_buffer
from display.hexboarddisplay import HexBoardDisplay
from mcts.mcts import MCTS
from nn.boardgamenetann import BoardGameNetANN
from nn.boardgamenetcnn import BoardGameNetCNN
from statemanager.hexstatemanager import HexStateManager


def rl_algorithm(nn, state_manager, display):
    """The reinforcement learning algorithm.

    Args:
        nn: the neural network class to use.
        state_manager (StateManager): the state manager class to use.
        display (GameBoardDisplay): game board display class to use.
    """
    # Configure logging level and format for console output
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    epsilon = config.EPSILON
    epsilon_decay = config.EPSILON_DECAY
    replay_buf = replay_buffer.ReplayBuffer(maxlen=config.REPLAY_BUFFER_SIZE)
    i_s = config.NUM_EPISODES // (config.TOPP_M - 1)  # Save interval
    replay_buf.clear()

    for g_a in tqdm(range(config.NUM_EPISODES + 1)):
        logging.info(f"Game {g_a}")
        root_state = state_manager.copy_state()

        mcts_tree = MCTS(root_state=root_state,
                         default_policy=nn, c=config.MTCS_C)
        s = mcts_tree.root

        winning_moves = None

        moves = 0
        while not s.state.check_winning_state():
            winning_moves = None
            # Check winning state, to end episode early
            if moves > (config.BOARD_SIZE - 1) * 2 - 1 and config.CHECK_WINNING_MOVES:
                winning_moves = s.state.get_winning_moves()

            print(f"Move {moves}")
            if config.MCTS_DYNAMIC_SIMS and winning_moves is None:
                start_time = time.time()
                i = 0

                while time.time() - start_time < 1 or i < config.MCTS_MIN_SIMULATIONS:
                    i += 1
                    node = mcts_tree.tree_search()
                    reward = mcts_tree.leaf_evaluation(node, epsilon)
                    mcts_tree.backpropagation(node, reward)

                print(f"Number of simulations: {i}")
            else:
                for _ in range(config.MTCS_SIMULATIONS + 1):
                    node = mcts_tree.tree_search()
                    reward = mcts_tree.leaf_evaluation(node, epsilon)
                    mcts_tree.backpropagation(node, reward)

            moves += 1
            if winning_moves is not None:
                print(f"Winning moves: {winning_moves}")

            distribution = (
                s.state.get_winning_distribution(winning_moves)
                if winning_moves is not None
                else s.state.get_visit_distribution(s)
            )

            replay_buf.add_case(
                (nn.convert_to_nn_input(s.state), distribution))

            s = (
                mcts_tree.select_winning_move(winning_moves[0])
                if winning_moves is not None
                else mcts_tree.select_best_distribution()
            )
            if config.DISPLAY_GAME_RL:
                display.display_board(s.state, delay=0.1, newest_move=s.move)
            mcts_tree.prune_tree(s)

        X, y = replay_buf.get_random_minibatch(config.BATCH_SIZE)

        nn.fit(X, y, epochs=config.NUM_EPOCHS)
        epsilon *= epsilon_decay

        if g_a % i_s == 0:
            nn.save_model(
                f"models/model_{config.BOARD_SIZE}x{config.BOARD_SIZE}_{g_a}")


if __name__ == "__main__":
    nn = BoardGameNetCNN(config.NEURAL_NETWORK_DIMENSIONS,
                         config.LEARNING_RATE,
                         config.ACTIVATION_FUNCTION,
                         config.OUTPUT_ACTIVATION_FUNCTION,
                         config.LOSS_FUNCTION,
                         config.ANN_OPTIMIZER,
                         config.BOARD_SIZE)
    state_manager = HexStateManager(config.BOARD_SIZE)
    display = HexBoardDisplay()
    rl_algorithm(nn=nn, state_manager=state_manager, display=display)
