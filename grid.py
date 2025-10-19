import numpy as np

class Grid:
    def __init__(self, gs):
        self.gs = gs
        self.hfences = np.zeros(shape=(gs, gs))
        self.vfences = np.zeros(shape=(gs, gs))

    def _addFence(self, fence):
        pass

    