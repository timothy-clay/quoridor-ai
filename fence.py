import pygame

class Fence:
    def __init__(self, orientation, col, row, color):
        """
        Create a fence object. 
        """

        self.orientation = orientation

        self.col = col
        self.row = row

        self.color = color
    
    def getCoords(self):
        """
        Get the coordinates (column, row, and orientation) of the fence. 
        """
        return self.col, self.row, self.orientation
    
    def getColor(self):
        """
        Get the color of the fence (for drawing purposes)
        """
        return self.color