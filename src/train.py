from mcts.mcts import MCTS
from statemanager.hexstatemanager import HexStateManager
from nn.boardgamenetcnn import BoardGameNetCNN
from display.hexboarddisplay import HexBoardDisplay
import config
import replay_buffer

from tqdm import tqdm
import time
import os
import logging

import threading
from concurrent.futures import ThreadPoolExecutor


def parallel_tree_search(mcts_tree, epsilon, lock, counter):
    start_time = time.time()
    while time.time() - start_time < 1 or counter[0] < config.MCTS_MIN_SIMULATIONS:
        with lock:
            node = mcts_tree.tree_search()

        reward = mcts_tree.leaf_evaluation(node, epsilon)

        with lock:
            mcts_tree.backpropagation(node, reward)
            counter[0] += 1

    return True


def rl_algorithm():
    # Configure logging level and format for console output
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    epsilon = config.EPSILON
    epsilon_decay = config.EPSILON_DECAY
    replay_buf = replay_buffer.ReplayBuffer(maxlen=config.REPLAY_BUFFER_SIZE)
    nn = BoardGameNetCNN(config.NEURAL_NETWORK_DIMENSIONS,
                         config.LEARNING_RATE,
                         config.ACTIVATION_FUNCTION,
                         config.OUTPUT_ACTIVATION_FUNCTION,
                         config.LOSS_FUNCTION,
                         config.ANN_OPTIMIZER,
                         config.BOARD_SIZE)
    i_s = config.SAVE_INTERVAL
    replay_buf.clear()

    for g_a in tqdm(range(config.NUM_EPISODES + 1)):
        logging.info(f"Game {g_a}")
        root_state = HexStateManager(config.BOARD_SIZE)

        mcts_tree = MCTS(root_state=root_state, default_policy=nn,
                         c=config.MTCS_C, verbose=config.MCTS_VERBOSE)
        s = mcts_tree.root

        winning_moves = None

        moves = 0
        while not s.state.check_winning_state():
            winning_moves = None
            # Check winning state, to end episode early
            if moves > (config.BOARD_SIZE - 1) * 2 - 1:
                winning_moves = s.state.has_winning_move()

            print(f"Move {moves}")
            if config.MCTS_DYNAMIC_SIMS and winning_moves is None:
                if config.MULTITHREAD_RL:
                    counter = [0]
                    lock = threading.Lock()

                    with ThreadPoolExecutor() as executor:
                        futures = [executor.submit(parallel_tree_search, mcts_tree, epsilon,
                                                   lock, counter) for _ in range(min(os.cpu_count(), 4))]

                        # Wait for all the futures to complete execution
                        for future in futures:
                            future.result()

                    print(f"Number of simulations: {counter[0]}")
                else:
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
                mcts_tree.get_winning_distribution(winning_moves)
                if winning_moves is not None
                else mcts_tree.get_visit_distribution()
            )

            replay_buf.add_case(
                (mcts_tree.root.state.convert_to_nn_input(), distribution))

            s = (
                mcts_tree.select_winning_move(winning_moves[0])
                if winning_moves is not None
                else mcts_tree.select_best_distribution()
            )
            if winning_moves is not None and len(winning_moves) > 1:
                pass
            # if config.DISPLAY_GAME_RL:
            #    display.display_board(s.state.convert_to_diamond_shape(
            #
            # ), delay=0.1, newest_move=s.move)
            mcts_tree.prune_tree(s)

        X, y = replay_buf.get_random_minibatch(config.BATCH_SIZE)

        nn.fit(X, y)
        epsilon *= epsilon_decay

        if g_a % i_s == 0:
            nn.save_model(
                f"models/model_{config.BOARD_SIZE}x{config.BOARD_SIZE}_{g_a}")


if __name__ == "__main__":
    rl_algorithm()
