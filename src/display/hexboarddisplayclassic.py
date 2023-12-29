import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
import numpy as np

from .gameboarddisplay import GameBoardDisplay


BLACK = (0, 0, 0)
BLUE = (0, 0, 1)
RED = (1, 0, 0)
WHITE = (1, 1, 1)
PX = 1 / plt.rcParams["figure.dpi"]


class HexBoardDisplayClassic(GameBoardDisplay):
    def __init__(self, width=700, height=700):
        self.init = False
        self.width = width
        self.height = height

    def display_board(
        self, state, delay=0, winner=None, newest_move=None, actor1=None, actor2=None
    ):
        """Displays the current state of the hex board.

        Args:
            state (list): current game state.
            delay (int, optional): pauses the execution to update the display. Defaults to 0.
            winner (tuple[int, int], optional): winner of the current state if there is one. Defaults to None.
            newest_move (tuple[int, int]), optional): the move played that lead to updating the display. Defaults to None.
        """
        if not self.init:
            self.fig = plt.figure(
                figsize=(
                    self.width * PX,
                    self.height * PX,   
                ),
                num="Hex",
            )
            self.ax = self.fig.add_subplot(111)

            self.init = True

        self.ax.clear()
        self.ax.set_axis_off()
        
        self.ax.set_aspect("equal")

        board = state.board
        
        hexagon_radius = 1

        # Horizontal spacing between each Hex cell (Cartesian):
        horizontal_spacing = 2 * hexagon_radius * np.cos(np.radians(30))
        # Vertical spacing between each Hex Cell.
        vertical_spacing = hexagon_radius * 1.5
        # This refers to the offset factor that is use to calculate the initial position X for each new row.
        horizontal_offset_factor = hexagon_radius * np.cos(np.radians(30))

        player1 = "Player 1" if actor1 is None else actor1
        player2 = "Player 2" if actor2 is None else actor2
        # Plots the legend that shows which color is which player.
        self.ax.plot(
            [],
            [],
            "o",
            markersize=10,
            color=RED,
            label=player1,
            markeredgecolor=(0, 0, 0),
            markeredgewidth=1,
        )
        self.ax.plot(
            [],
            [],
            "o",
            markersize=10,
            color=BLUE,
            label=player2,
            markeredgecolor=(0, 0, 0),
            markeredgewidth=1,
        )
        self.ax.plot(
            [],
            [],
            "o",
            markersize=10,
            color=WHITE,
            label="Unoccupied",
            markeredgecolor=(0, 0, 0),
            markeredgewidth=1,
        )

        # Add the legend to the plot
        self.ax.legend(loc="upper right", numpoints=1, fontsize=10)

        # Initial position.
        posY = 0
        startposX = 0

        for i in range(len(board)):
            # The initial position X is offset by half the horizontal spacing to center the board minus the offset factor and the length of the row.
            # This is because the initial offset needs to be smaller for longer rows.
            startposX += horizontal_offset_factor
            posX = startposX

            for j in range(len(board[i])):
                # Determines the color of the cell to be drawn.
                color = (
                    RED
                    if board[i][j] == 1
                    else BLUE
                    if board[i][j] == -1
                    else WHITE
                )
                
                # Plot zigzag lines to indicate the starting and ending positions.
                posX_left = posX - horizontal_spacing / 2
                posX_right = posX + horizontal_spacing / 2
                posX_center = posX
                
                posY_upper_middle = posY + vertical_spacing / 3
                posY_top = posY + hexagon_radius
                posY_bottom = posY - hexagon_radius
                posY_lower_middle = posY - vertical_spacing / 3
                
                if j == 0:
                    self.ax.plot(
                        [posX_left, posX_left, posX_center],
                        [posY_upper_middle, posY_lower_middle, posY_bottom],
                        "-", color=BLUE, linewidth=2,
                    )
                elif j == len(board[i]) - 1:
                    self.ax.plot(
                        [posX_center, posX_right, posX_right],
                        [posY_top, posY_upper_middle, posY_lower_middle],
                        "-", color=BLUE, linewidth=2,
                    )
                
                if i == 0:
                    self.ax.plot(
                        [posX_left, posX_center, posX_right],
                        [posY_upper_middle, posY_top, posY_upper_middle],
                        "-", color=RED, linewidth=2,
                    )
                elif i == len(board) - 1:
                    self.ax.plot(
                        [posX_left, posX_center, posX_right],
                        [posY_lower_middle, posY_bottom, posY_lower_middle],
                        "-", color=RED, linewidth=2,
                    )
                
                if newest_move and newest_move == (i, j):
                    hex_patch = RegularPolygon(
                        (posX, posY),
                        numVertices=6,
                        radius=hexagon_radius,
                        orientation=0,
                        facecolor=color,
                        edgecolor=BLACK,
                        alpha=1.0,
                    )
                else:
                    hex_patch = RegularPolygon(
                        (posX, posY),
                        numVertices=6,
                        radius=hexagon_radius,
                        orientation=0,
                        facecolor=color,
                        edgecolor=BLACK,
                        alpha=0.8,
                    )
                
                letter = chr(ord("A") + j)
                num_pos = f"{letter}{i + 1}"
                # This just enlarges the newest node, to make it easier to see what moves are taken.
                self.ax.add_patch(hex_patch)
                self.ax.text(posX, posY, num_pos, ha="center", va="center", fontsize=70/len(board))
                # Increase X by horizontal spacing.
                posX += horizontal_spacing

            # Increase Y by vertical spacing after each row is complete.
            posY -= vertical_spacing

        title = "Hex" if not state.switched else "Hex (Switched)"
        plt.title(title, fontsize=20)

        if winner is not None:
            # Update title to the winner if there is one.
            plt.title(f"The winner is player {1 if winner == 1 else 2}", fontsize=20)
            plt.draw()
            plt.pause(delay)

        # If the delay is greater than 0, it means we want to update the display, and if not we want to only show one display
        if delay > 0:
            plt.draw()
            plt.pause(delay)
