import pygame

class Fence:
    def __init__(self, orientation, col, row, color):
        self.orientation = orientation

        self.col = col
        self.row = row

        self.color = color
    
    def _getCoords(self):
        return self.col, self.row, self.orientation
    
    def _getColor(self):
        return self.color