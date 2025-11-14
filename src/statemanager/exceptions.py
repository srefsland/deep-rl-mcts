class IllegalMoveException(Exception):
    def __init__(self, message="Attempted to make an illegal move"):
        self.message = message
        super().__init__(self.message)
