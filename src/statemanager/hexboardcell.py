class HexBoardCell:
    """The cells that the hex board consists of.
    """
    def __init__(self, x, y):
        self.position = (x, y)
        self.occupant = (0, 0)

    def is_empty(self):
        """Checks if the cell is empty.

        Returns:
            bool: true if occupeant is (0, 0), false otherwise.
        """
        return self.occupant == (0, 0)
