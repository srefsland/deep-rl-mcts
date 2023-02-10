class BoardCell:
    def __init__(self, x, y):
        self._pos = (x, y)
        self._owner = (0, 0)

    def set_position(self, x, y):
        self._pos = (x, y)

    def get_position(self):
        return self._pos

    def set_owner(self, owner):
        self._owner = owner

    def get_owner(self):
        return self._owner
