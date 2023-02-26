class HexBoardCell:
    def __init__(self, x, y):
        self.position = (x, y)
        self.occupant = (0, 0)

    def is_empty(self):
        return self.occupant == (0, 0)
