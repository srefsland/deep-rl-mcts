# AlphaGo Knock-Off Project

![test](https://github.com/simenrefsland/it3105-artificial-intelligence-programming/actions/workflows/python-ci.yml/badge.svg)

<img src="https://github.com/simenrefsland/it3105-artificial-intelligence-programming/blob/master/.github/hex.gif" width=500px>

# Ideas for implementation
- For MCTS rollouts, check for immediate winning moves (after n moves in a nxn board) and play them (ignoring the default policy). Should boost the speed of the algorithm by providing a better distribution for the anet to train on.
- Rules for expanding nodes during tree search, expand leaf only on N visits maybe?
- Look for more efficient representations of the board state and such, usage of deque is more efficient than list for example in the replay buffer.
