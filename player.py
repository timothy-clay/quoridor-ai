import pygame
from pygame import gfxdraw
from collections import deque
import numpy as np
import heapq

from fence import *

DIRECTIONS = {'w':[0, -1], 's':[0, 1], 'a':[-1, 0], 'd':[1, 0]}

class Player:
    def __init__(self, name, col, row, color, gridSize, fences=None, target_row=None):

        self.name = name

        self.col = col
        self.row = row

        self.raw_color = color.lstrip('#')
        r = int(self.raw_color[0:2], 16)
        g = int(self.raw_color[2:4], 16)
        b = int(self.raw_color[4:6], 16)

        self.color = (r, g, b)

        self.gridSize = gridSize

        if fences is not None:
            self.fences = fences
        else:
            self.fences = set()

        if target_row is not None:
            self.target_row = target_row
        else:
            self.target_row = self.gridSize - self.row - 1

    def duplicate(self):
        return Player(self.name, self.col, self.row, self.raw_color, self.gridSize, self.fences.copy(), self.target_row)
    
    def getValidTurns(self, grid, opponent):
        valid_turns = []

        for movement_direction in 'wsad':
            col_change, row_change = DIRECTIONS[movement_direction]

            if self._canMove(grid, col_change, row_change):
                active_col, active_row = self._getCoords()
                
                if grid._isPawn(active_col + col_change, active_row + row_change):
                    if self._canMove(grid, col_change * 2, row_change * 2):
                        valid_turns.append(f'p{movement_direction}')
                else:
                    valid_turns.append(f'p{movement_direction}')

        if self._getRemainingFences() > 0:
            
            for orientation in 'hv':
                for col in 'abcdefghi':
                    for row in '123456789':

                        fence_col = 'abcdefghi'.index(col)
                        fence_row = '987654321'.index(row)
                        
                        if self._canPlaceFence(grid, opponent, orientation, fence_col, fence_row):
                            valid_turns.append(f'f{orientation}{col}{row}')

        return valid_turns
    

    def _checkWin(self):
        return self.row == self.target_row
    
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
        fence = Fence(orientation, col, row, self.color)
        self.fences.add(fence)
        grid._addFence(fence)
        return self
    
    def _getRemainingFences(self):
        return 10 - len(self.fences)
    

    def _canPlaceFence(self, grid, opponent, orientation, col, row):

        temp_fence = Fence(orientation, col, row, self.color)

        if not grid._validFencePlacement(temp_fence):
            return False
        
        temp_hfences, temp_vfences = grid._testFencePlacement(temp_fence)

        if not self._checkValidPath(grid, temp_vfences, temp_hfences):
            return False
        
        if not opponent._checkValidPath(grid, temp_vfences, temp_hfences):
            return False
        
        return True
    
    def _checkValidPath(self, grid, vfences=None, hfences=None):

        if self._getShortestPath(grid, vfences, hfences) >= 0:
            return True 
        

    
    def _getShortestPath(self, grid, vfences=None, hfences=None):

        
        start = (self.col, self.row)

        open_heap = []
        heapq.heappush(open_heap, (np.abs(self.target_row - self.row), 0, start))

        visited_cells = set()
        distances = {start: 0}

        while open_heap:

            total_estimated_distance, current_distance, (current_col, current_row) = heapq.heappop(open_heap)

            if current_row == self.target_row:
                return current_distance

            if (current_col, current_row) in visited_cells:
                continue
            visited_cells.add((current_col, current_row))
            
            for col_change, row_change in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                new_col = current_col + col_change
                new_row = current_row + row_change
                new_coords = (new_col, new_row)

                if new_coords in visited_cells:
                    continue

                if grid._validMove(current_col, current_row, col_change, row_change, vfences, hfences):
                    new_distance = current_distance + 1
                    if new_distance < distances.get(new_distance, float('inf')):
                        distances[new_coords] = new_distance
                        total_estimated_distance = new_distance + np.abs(self.target_row - new_row)
                        heapq.heappush(open_heap, (total_estimated_distance, new_distance, new_coords))

        return -1

    