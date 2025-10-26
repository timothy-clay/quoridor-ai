import pygame
from pygame import gfxdraw
from collections import deque

from fence import *

class Player:
    def __init__(self, name, col, row, color, game, fences=None, target_row=None, radius=None):

        self.name = name

        self.col = col
        self.row = row

        color = color.lstrip('#')
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)

        self.color = (r, g, b)
        self.game = game

        if fences:
            self.fences = fences
        else:
            self.fences = set()

        if target_row:
            self.target_row = target_row
        else:
            self.target_row = self.game.gridSize - self.row - 1

        if radius:
            self.radius = radius
        else:
            self.radius = int(self.game.cellSize * 0.4)

    def copy(self):
        return Player(self.name, self.col, self.row, self.color, self.game, self.fences, self.target_row, self.radius)

    def _checkWin(self):
        return self.row == self.target_row

    def _drawPawn(self, screen):
        x_pixels, y_pixels = self.game.getPawnPixels(self.col, self.row)
        gfxdraw.aacircle(screen, x_pixels, y_pixels, self.radius, self.color)
        gfxdraw.filled_circle(screen, x_pixels, y_pixels, self.radius, self.color)
        return
    
    def _getName(self):
        return self.name
    
    def _getCoords(self):
        return self.col, self.row
    
    def _getColor(self):
        return self.color

    def _movePawn(self, grid, col_change, row_change):
        grid._movePawn(self.col, self.row, col_change, row_change)
        self.col += col_change
        self.row += row_change
        return self
    
    def _canMove(self, grid, col_change, row_change):
        return grid._validMove(self.col, self.row, col_change, row_change)
    
    def _placeFence(self, grid, orientation, col, row):
        fence = Fence(orientation, col, row, self.color, self.game)
        self.fences.add(fence)
        grid._addFence(fence)
        return self
    
    def _getRemainingFences(self):
        return 10 - len(self.fences)
    

    def _canPlaceFence(self, grid, opponent, orientation, col, row):

        temp_fence = Fence(orientation, col, row, self.color, self.game)

        if not grid._validFencePlacement(temp_fence):
            return False
        
        temp_hfences, temp_vfences = grid._testFencePlacement(temp_fence)

        if not self._checkValidPath(grid, temp_vfences, temp_hfences):
            return False
        
        if not opponent._checkValidPath(grid, temp_vfences, temp_hfences):
            return False
        
        return True
    
    def _checkValidPath(self, grid, vfences, hfences):

        if self._getShortestPath(grid, vfences, hfences) >= 0:
            return True 
    
    def _getShortestPath(self, grid, vfences, hfences):
        visited_cells = set()
        cells_to_visit = deque()

        cells_to_visit.append((self.col, self.row, 0))

        while len(cells_to_visit) > 0:
            current_col, current_row, distance = cells_to_visit.popleft()
            visited_cells.add((current_col, current_row))

            if current_row == self.target_row:
                return distance
            
            for col_change, row_change in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_col = current_col + col_change
                new_row = current_row + row_change
                if (new_col, new_row) not in visited_cells:
                    if grid._validMove(current_col, current_row, col_change, row_change, vfences, hfences):
                        cells_to_visit.append((new_col, new_row, distance + 1))

        return -1