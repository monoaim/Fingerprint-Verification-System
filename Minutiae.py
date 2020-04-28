# minutiae types
M_TYPE_UNKNOWN = 0
M_TYPE_ENDPOINT = 1
M_TYPE_BIFURCATION = 2


class Minutiae:

    def __init__(self, x, y, type=M_TYPE_UNKNOWN):
        self.x = x
        self.y = y
        self.type = type

    def getType(self):
        return self.type

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def __str__(self):
        return "({}, {})".format(self.x, self.y)
