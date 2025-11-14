# Deep RL MCTS with Hex

This project implements a simplified version of the AlphaGo/AlphaZero architecture that has been successful when applied to board games such as Chess and Go. To create a more manageable state space, the neural network has been trained for the game Hex. The approach combines Deep Learning (DL) and Reinforcement Learning (RL) using On-Policy Monte Carlo Tree Search (MCTS), which creates distribution targets through self-play, which the neural network can use to improve its playing strength. These targets are created by initially playing completely random moves, and gradually increasing the probability of using the neural network to make moves, which is trained after each episode.

The neural network is a Convolutional Neural Network (CNN), which outputs both the actor and critic evaluation, where the actor's output is used to make moves, and the critic is used to evaluate the current position. The purpose of the critic is to reduce the number of rollouts when reaching a leaf node.

## Demonstration

<img src="https://github.com/simenrefsland/it3105-artificial-intelligence-programming/blob/master/.github/images/hex.gif" width=500px>
