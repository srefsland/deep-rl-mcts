import matplotlib.pyplot as plt
from .hexboard import HexBoard

BLACK = (0, 0, 0)
RED = (1, 0, 0)
WHITE = (1, 1, 1)
PX = 1/plt.rcParams['figure.dpi']


class BoardDisplay:
    def __init__(self, width=1000, height=800, board_skewness_factor=0.75):
        self.width = width
        self.height = height
        self.board_skewness_factor = board_skewness_factor
        self.fig = plt.figure(
            figsize=(width*board_skewness_factor*PX, height*PX), num="Hex")
        self.ax = self.fig.add_subplot(111)

    def visualize(self, board, delay=0, winner=None):
        self.ax.clear()
        self.ax.set_axis_off()

        # Board is reversed because the board is drawn from the bottom left corner
        board.reverse()

        horizontal_spacing = (self.width/len(board)-1) * \
            (2 * self.board_skewness_factor)
        vertical_spacing = (self.height/len(board)-1)
        horizontal_offset_factor = (
            self.width/len(board)-1) * (self.board_skewness_factor)

        # TODO: Find a better way to scale the circles. This is just some
        # random stuff I experimented with
        circle_radius = (horizontal_spacing)**0.90 / 5

        self.ax.plot([], [], 'o', markersize=10, color=RED,
                     label='Player 1', markeredgecolor=(0, 0, 0), markeredgewidth=1)
        self.ax.plot([], [], 'o', markersize=10, color=BLACK,
                     label='Player 2', markeredgecolor=(0, 0, 0), markeredgewidth=1)
        self.ax.plot([], [], 'o', markersize=10, color=WHITE,
                     label='Empty cell', markeredgecolor=(0, 0, 0), markeredgewidth=1)

        # Add the legend to the plot
        self.ax.legend(loc='upper left', numpoints=1, fontsize=10)

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

                color = RED if board[i][j].get_owner() == (
                    1, 0) else BLACK if board[i][j].get_owner() == (0, 1) else WHITE

                self.ax.plot(posX, posY, 'o', markersize=circle_radius,
                             markeredgecolor=(0, 0, 0),
                             markerfacecolor=color,
                             markeredgewidth=1)

                posX += horizontal_spacing

            posY += vertical_spacing

        plt.title("Hex", fontsize=20)

        if winner is not None:
            plt.title(
                f'The winner is player {1 if winner == (1, 0) else 2}', fontsize=20)
            self.fig.canvas.draw()

        # If the delay is greater than 0, it means we want to update the display, and if not we want to only show one display
        if delay > 0:
            self.fig.canvas.draw()
            plt.pause(delay)
        else:
            plt.show()
