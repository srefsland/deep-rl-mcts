class HexNode:
    def __init__(self, x, y):
        self.pos = (x, y)
        self.owner = (0, 0)
                
    def set_position(self, x, y):
        self.pos = (x, y)
        
    def set_owner(self, owner):
        self.owner = owner
        
    def get_owner(self):
        if self.owner == (0, 0):
            return 0
        elif self.owner == (1, 0):
            return 1
        else:
            return 2