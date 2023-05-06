import logging
import time

import numpy as np
from tqdm import tqdm

import config
import replay_buffer
from display.hexboarddisplay import HexBoardDisplay
from mcts.mcts import MCTS
from nn.boardgamenetann import BoardGameNetANN
from statemanager.hexstatemanager import HexStateManager


def rl_algorithm(nn, state_manager, display):
    """The reinforcement learning algorithm.

    Args:
        nn: the neural network class to use.
        state_manager (StateManager): the state manager class to use.
        display (GameBoardDisplay): game board display class to use.
    """
    # Configure logging level and format for console output
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    epsilon = config.EPSILON
    epsilon_decay = config.EPSILON_DECAY
    epsilon_critic = config.EPSILON_CRITIC
    epsilon_decay_critic = config.EPSILON_DECAY_CRITIC
    replay_buf = replay_buffer.ReplayBuffer(maxlen=config.REPLAY_BUFFER_SIZE)
    i_s = config.NUM_EPISODES // (config.TOPP_M - 1)  # Save interval
    replay_buf.clear()

    for g_a in tqdm(range(config.NUM_EPISODES + 1)):
        logging.info(f"Game {g_a}")

        state_manager.reset()

        mcts_tree = MCTS(
            state_manager=state_manager, default_policy=nn, c=config.MTCS_C
        )
        s = mcts_tree.root

        winning_moves = None

        moves = 0
        while not mcts_tree.state_manager.check_winning_state():
            winning_moves = None
            # Check winning state, to end episode early
            if (
                moves > (config.BOARD_SIZE - 1) * 2 - 1
                and config.CHECK_WINNING_MOVES_RL
            ):
                winning_moves = mcts_tree.state_manager.get_winning_moves()

            print(f"Move {moves}")
            if config.MCTS_DYNAMIC_SIMS and winning_moves is None:
                start_time = time.time()
                i = 0

                while (
                    time.time() - start_time < config.MCTS_DYNAMIC_SIMS_TIME
                    or i < config.MCTS_MIN_SIMULATIONS
                ):
                    i += 1
                    node = mcts_tree.tree_search()
                    reward = mcts_tree.leaf_evaluation(node, epsilon, epsilon_critic)
                    mcts_tree.backpropagation(node, reward)

                print(f"Number of simulations: {i}")
            else:
                for _ in range(config.MTCS_SIMULATIONS + 1):
                    node = mcts_tree.tree_search()
                    reward = mcts_tree.leaf_evaluation(node, epsilon, epsilon_critic)
                    mcts_tree.backpropagation(node, reward)

            moves += 1
            if winning_moves is not None:
                print(f"Winning moves: {winning_moves}")

            distribution = (
                mcts_tree.state_manager.get_winning_distribution(winning_moves)
                if winning_moves is not None
                else mcts_tree.state_manager.get_visit_distribution(s)
            )

            replay_buf.add_case(
                (
                    nn.convert_to_nn_input(s.state, s.player),
                    distribution,
                    np.array([s.q]),
                )
            )

            s = (
                mcts_tree.select_winning_move(winning_moves[0])
                if winning_moves is not None
                else mcts_tree.select_best_distribution()
            )

            mcts_tree.state_manager.update_state(s.state, s.player)

            if config.DISPLAY_GAME_RL and g_a % config.DISPLAY_GAME_RL_INTERVAL == 0:
                display.display_board(
                    mcts_tree.state_manager, delay=0.1, newest_move=s.move
                )
            mcts_tree.prune_tree(s)

        X, y_actor, y_critic = replay_buf.get_random_minibatch(config.BATCH_SIZE)

        nn.fit(X, y_actor, y_critic, epochs=config.NUM_EPOCHS)
        epsilon *= epsilon_decay
        epsilon_critic *= epsilon_decay_critic

        if g_a % i_s == 0:
            nn.save_model(f"models/model_{config.BOARD_SIZE}x{config.BOARD_SIZE}_{g_a}")

            if g_a != 0:
                nn.display_losses()


if __name__ == "__main__":
    nn = BoardGameNetANN(
        config.NEURAL_NETWORK_DIMENSIONS,
        config.LEARNING_RATE,
        config.ACTIVATION_FUNCTION,
        config.OUTPUT_ACTIVATION_FUNCTION_ACTOR,
        config.OUTPUT_ACTIVATION_FUNCTION_CRITIC,
        config.LOSS_FUNCTION_ACTOR,
        config.LOSS_FUNCTION_CRITIC,
        config.ANN_OPTIMIZER,
        config.BOARD_SIZE,
    )
    state_manager = HexStateManager(config.BOARD_SIZE)
    display = HexBoardDisplay()
    rl_algorithm(nn=nn, state_manager=state_manager, display=display)
