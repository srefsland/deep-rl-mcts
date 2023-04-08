from mcts.mcts import MCTS
from statemanager.hexstatemanager import HexStateManager
from nn.boardgamenetcnn import BoardGameNetCNN
import config
import replay_buffer

import cProfile
import pstats
from tqdm import tqdm
import time
import os

import threading
from concurrent.futures import ThreadPoolExecutor


def parallel_tree_search(mcts_tree, epsilon, lock, counter):
    start_time = time.time()
    while time.time() - start_time < 1 or counter[0] < 200:
        with lock:
            node = mcts_tree.tree_search()

        reward = mcts_tree.leaf_evaluation(node, epsilon)

        with lock:
            mcts_tree.backpropagation(node, reward)
            counter[0] += 1

    return True


def rl_algorithm():
    profiler = cProfile.Profile()
    profiler.enable()

    epsilon = config.EPSILON
    epsilon_decay = config.EPSILON_DECAY
    replay_buf = replay_buffer.ReplayBuffer()
    nn = BoardGameNetCNN(config.NEURONS_PER_LAYER,
                         config.NEURONS_PER_LAYER,
                         config.LEARNING_RATE,
                         config.ACTIVATION_FUNCTION,
                         config.OUTPUT_ACTIVATION_FUNCTION,
                         config.LOSS_FUNCTION,
                         config.ANN_OPTIMIZER,
                         config.BOARD_SIZE)
    i_s = config.SAVE_INTERVAL
    replay_buf.clear()

    for g_a in tqdm(range(config.NUM_EPISODES + 1)):
        print(f"Game {g_a}")
        root_state = HexStateManager(config.BOARD_SIZE)

        mcts_tree = MCTS(root_state=root_state, default_policy=nn,
                         c=config.MTCS_C, verbose=config.MCTS_VERBOSE)
        s = mcts_tree.root

        while not s.state.check_winning_state():
            if config.MCTS_DYNAMIC_SIMS:
                # start_time = time.time()
                # i = 0

                # while time.time() - start_time < 1 or i < 50:
                #     i += 1
                #     node = mcts_tree.tree_search()
                #     reward = mcts_tree.leaf_evaluation(node, epsilon)
                #     mcts_tree.backpropagation(node, reward)

                # print(f"Number of simulations: {i}")
                counter = [0]
                lock = threading.Lock()

                with ThreadPoolExecutor() as executor:
                    futures = [executor.submit(parallel_tree_search, mcts_tree, epsilon,
                                               lock, counter) for _ in range(os.cpu_count())]

                    # Wait for all the futures to complete execution
                    for future in futures:
                        future.result()

                print(f"Number of simulations: {counter[0]}")
            else:
                for g_s in range(config.MTCS_SIMULATIONS + 1):
                    node = mcts_tree.tree_search()
                    reward = mcts_tree.leaf_evaluation(node, epsilon)
                    mcts_tree.backpropagation(node, reward)

            D = mcts_tree.get_visit_distribution()
            replay_buf.add_case(
                (mcts_tree.root.state.convert_to_nn_input(), D))
            s = mcts_tree.select_best_distribution()
            mcts_tree.prune_tree(s)

        X, y = replay_buf.get_random_minibatch(config.BATCH_SIZE)

        nn.fit(X, y)
        epsilon *= epsilon_decay

        if g_a % i_s == 0:
            nn.save_model(
                f"models/model_{config.BOARD_SIZE}x{config.BOARD_SIZE}_{g_a}")

    profiler.disable()
    stats_filename = "mcts_simulation_stats.prof"
    profiler.dump_stats(stats_filename)


rl_algorithm()
stats = pstats.Stats("mcts_simulation_stats.prof")
stats.strip_dirs().sort_stats("cumtime").print_stats("boardgamenetcnn.py")
