import numpy as np

class Grid:
    def __init__(self, gs):
        self.gs = gs
        self.hfences = np.zeros(shape=(gs, gs))
        self.vfences = np.zeros(shape=(gs, gs))

    def _addFence(self, fence):

        col, row, orientation = fence._getCoords()

        if orientation == 'h':
            self.hfences[row, col:col+2] = 1

        elif orientation == 'v':
            self.vfences[row:row+2, col] = 1


    def _getHFences(self):
        return self.hfences
    
    def _getVFences(self):
        return self.vfences

        

    