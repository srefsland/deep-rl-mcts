import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

from .gameboarddisplay import GameBoardDisplay
import numpy as np

BLACK = (0, 0, 0)
RED = (1, 0, 0)
WHITE = (1, 1, 1)
PX = 1/plt.rcParams['figure.dpi']


class HexBoardDisplay(GameBoardDisplay):
    def __init__(self, width=1000, height=800, board_skewness_factor=0.75):
        self.width = width
        self.height = height
        self.board_skewness_factor = board_skewness_factor
        self.fig = plt.figure(
            figsize=(width*board_skewness_factor*PX, height*PX), num="Hex")
        self.ax = self.fig.add_subplot(111)

    def display_board(self, state, delay=0, winner=None, newest_move=None, actor1=None, actor2=None):
        """Displays the current state of the hex board.

        Args:
            state (list): current game state.
            delay (int, optional): pauses the execution to update the display. Defaults to 0.
            winner (tuple[int, int], optional): winner of the current state if there is one. Defaults to None.
            newest_move (tuple[int, int]), optional): the move played that lead to updating the display. Defaults to None.
        """
        self.ax.clear()
        self.ax.set_axis_off()

        board = self._convert_to_display_shape(state.board)

        # Horizontal spacing between each Hex cell.
        horizontal_spacing = (self.width/len(board)-1) * \
            (2 * self.board_skewness_factor)
        # Vertical spacing between each Hex Cell.
        vertical_spacing = (self.height/len(board)-1)
        # This refers to the offset factor that is use to calculate the initial position X for each new row.
        horizontal_offset_factor = (
            self.width/len(board)-1) * (self.board_skewness_factor)

        # TODO: Find a better way to scale the circles. This is just some
        # random stuff I experimented with, but should work for 3x3 to 9x9.
        circle_radius = (horizontal_spacing)**0.90 / 5

        player1 = "Player 1" if actor1 is None else actor1
        player2 = "Player 2" if actor2 is None else actor2
        # Plots the legend that shows which color is which player.
        self.ax.plot([], [], 'o', markersize=10, color=RED,
                     label=player1, markeredgecolor=(0, 0, 0), markeredgewidth=1)
        self.ax.plot([], [], 'o', markersize=10, color=BLACK,
                     label=player2, markeredgecolor=(0, 0, 0), markeredgewidth=1)
        self.ax.plot([], [], 'o', markersize=10, color=WHITE,
                     label='Unoccupied', markeredgecolor=(0, 0, 0), markeredgewidth=1)

        # Add the legend to the plot
        self.ax.legend(loc='upper left', numpoints=1, fontsize=10)

        # Initial position.
        posY = vertical_spacing / 2

        for i in range(len(board)):
            # The initial position X is offset by half the horizontal spacing to center the board minus the offset factor and the length of the row.
            # This is because the initial offset needs to be smaller for longer rows.
            posX = self.width / 2 - \
                (horizontal_offset_factor * (len(board[i]) - 1))

            for j in range(len(board[i])):
                # Connect to the right circle if the circle isn't the last one in the row
                if j < len(board[i]) - 1:
                    posX_right = posX + horizontal_spacing
                    posY_right = posY
                    self.ax.plot([posX, posX_right], [
                                 posY, posY_right], '-', color='black')

                # Connect to the bottom right circle if we are not on the last row and if the length of the next row is greater than the current circle index
                if i < len(board) - 1 and j < len(board[i+1]):
                    posX_bottom_right = posX + horizontal_spacing / 2
                    posY_bottom_right = posY + vertical_spacing
                    self.ax.plot([posX, posX_bottom_right], [
                                 posY, posY_bottom_right], '-', color='black')

                # Connect to the bottom left circle if we are not on the last row and if the next row is shorter, skip the first circle
                if i < len(board) - 1 and (j > 0 or len(board[i]) < len(board[i+1])):
                    # This is because the distance between nodes on different rows are half what they are on the same row
                    posX_bottom_left = posX - horizontal_spacing / 2
                    posY_bottom_left = posY + vertical_spacing
                    self.ax.plot([posX, posX_bottom_left], [
                                 posY, posY_bottom_left], '-', color='black')

                # Determines the color of the cell to be drawn.
                color = RED if board[i][j].occupant == (
                    1, 0) else BLACK if board[i][j].occupant == (0, 1) else WHITE

                # This just enlarges the newest node, to make it easier to see what moves are taken.
                if newest_move is not None and newest_move == board[i][j].position:
                    self.ax.plot(posX, posY, 'o', markersize=circle_radius*1.15,
                                 markeredgecolor=(0, 0, 0),
                                 markerfacecolor=color,
                                 markeredgewidth=1.5)
                # Normal size if not.
                else:
                    self.ax.plot(posX, posY, 'o', markersize=circle_radius,
                                 markeredgecolor=(0, 0, 0),
                                 markerfacecolor=color,
                                 markeredgewidth=1)
                # Increase X by horizontal spacing.
                posX += horizontal_spacing

            # Increase Y by vertical spacing after each row is complete.
            posY += vertical_spacing

        plt.title("Hex", fontsize=20)

        if winner is not None:
            # Update title to the winner if there is one.
            plt.title(
                f'The winner is player {1 if winner == (1, 0) else 2}', fontsize=20)
            plt.draw()
            plt.pause(delay)

        # If the delay is greater than 0, it means we want to update the display, and if not we want to only show one display
        if delay > 0:
            plt.draw()
            plt.pause(delay)

    def _convert_to_display_shape(self, board):
        """Converts the game state to a diamond shape, which is the input format for displaying the board.
        In effect, this rotates the board 45 degrees clockwise.

        Example:
        [[1],
        [1, 2],
        [1, 2, 3],
        [1, 2],
        [1]]

        Returns:
            list: the converted board state.
        """
        board_size = len(board)
        diamond_array = []
        board = np.flipud(board)
        for i in range(board_size - 1, -board_size, -1):
            diamond_array.append(np.diagonal(board, i).tolist())

        return diamond_array
