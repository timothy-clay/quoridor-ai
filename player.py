import pygame
from pygame import gfxdraw

from fence import *

class Player:
    def __init__(self, col, row, color, game):
        self.col = col
        self.row = row

        color = color.lstrip('#')
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)

        self.color = (r, g, b)
        self.game = game

        self.fences = set()

        self.radius = int(self.game.cellSize * 0.4)

    def _drawPawn(self, screen):
        x_pixels, y_pixels = self.game.getPawnPixels(self.col, self.row)
        gfxdraw.aacircle(screen, x_pixels, y_pixels, self.radius, self.color)
        gfxdraw.filled_circle(screen, x_pixels, y_pixels, self.radius, self.color)
        return
    
    def _getCoords(self):
        return self.col, self.row

    def _movePawn(self, grid, col_change, row_change):
        grid._movePawn(self.col, self.row, col_change, row_change)
        self.col += col_change
        self.row += row_change
        return self
    
    def _canMove(self, grid, col_change, row_change):

        if self.col + col_change >= self.game.gridSize or self.row + row_change >= self.game.gridSize:
            return False
        
        elif self.col + col_change < 0 or self.row + row_change < 0:
            return False

        if col_change != 0:

            vfences = grid._getVFences()

            # check for vertical fences
            if col_change > 0:

                # check for fence on new cell
                if vfences[self.row, self.col + col_change] == 1:
                    return False
                
            elif col_change < 0:
                
                # check for fence on current cell
                if vfences[self.row, self.col + col_change + 1] == 1:
                    return False

        elif row_change != 0:

            hfences = grid._getHFences()

            # check for horizontal fences
            if row_change > 0:
                
                # check for fence on new cell
                if hfences[self.row + row_change, self.col] == 1:
                    return False

            elif row_change < 0:
                
                # check for fence on current cell
                if hfences[self.row + row_change + 1, self.col] == 1:
                    return False
        
        return True
    
    def _placeFence(self, grid, orientation, col, row):
        fence = Fence(orientation, col, row, self.color, self.game)
        self.fences.add(fence)
        grid._addFence(fence)
        return self

    def _canPlaceFence(self, grid, orientation, col, row):
        return True
