import pygame
from pygame import gfxdraw
from collections import deque
import numpy as np
import heapq

from fence import *

DIRECTIONS = {'w':[0, -1], 's':[0, 1], 'a':[-1, 0], 'd':[1, 0]}

class Player:
    def __init__(self, name, col, row, color, gridSize, fences=None, target_row=None):
        """
        Creates a player object that stores mostly pawn information.
        """

        # player name
        self.name = name

        # starting position
        self.col = col
        self.row = row

        # color 
        self.raw_color = color.lstrip('#')
        r = int(self.raw_color[0:2], 16)
        g = int(self.raw_color[2:4], 16)
        b = int(self.raw_color[4:6], 16)
        self.color = (r, g, b)

        # size of game grid
        self.gridSize = gridSize

        # load fences and target row if predefined, else initialize
        self.fences = fences if fences is not None else set()
        self.target_row = target_row if target_row is not None else self.gridSize - self.row - 1

        # store number of cell visits and the previous move, both as a string and as a vector
        self.visited_counts = np.zeros(shape=(gridSize, gridSize))
        self.prev_move = None
        self.prev_move_onehot = np.zeros(5)


    def duplicate(self):
        """
        Creates a copy of the player object with the same position, target, and fences.
        """
        return Player(name=self.name, 
                      col=self.col, 
                      row=self.row, 
                      color=self.raw_color, 
                      gridSize=self.gridSize, 
                      fences=self.fences.copy(), 
                      target_row=self.target_row)
    

    def getValidTurns(self, grid, opponent):
        """
        Get a list of all possible turns a player can take from their current position / game state. 
        """

        # store moves
        valid_turns = []

        # check movement options
        for movement_direction in 'wsad':
            col_change, row_change = DIRECTIONS[movement_direction]

            # check initial move
            if self._canMove(grid, col_change, row_change):
                active_col, active_row = self.getCoords()
                
                # check if pawn jump is needed / possible
                if grid._isPawn(active_col + col_change, active_row + row_change):
                    if self._canMove(grid, col_change * 2, row_change * 2):
                        valid_turns.append(f'p{movement_direction}')
                else:
                    valid_turns.append(f'p{movement_direction}')

        # check fence placement options
        if self.getRemainingFences() > 0:
            
            # loop through each possible fence
            for orientation in 'hv':
                for col in 'abcdefghi':
                    for row in '123456789':

                        fence_col = 'abcdefghi'.index(col)
                        fence_row = '987654321'.index(row)
                        
                        # check if the placement is valid and store move if so
                        if self._canPlaceFence(grid, opponent, orientation, fence_col, fence_row):
                            valid_turns.append(f'f{orientation}{col}{row}')

        return valid_turns
    
    def checkMoveValidity(self, game, grid, opponent, move):
        """
        Get a list of all possible turns a player can take from their current position / game state. 
        """

        valid_move = False

        if move[0] == 'p':
            if move[1] in 'wsad':
                movement_direction = move[1]
                col_change, row_change = DIRECTIONS[movement_direction]
                if self._canMove(grid, col_change, row_change):
                    active_col, active_row = self.getCoords()
                    
                    # check if pawn jump is needed / possible
                    if grid._isPawn(active_col + col_change, active_row + row_change):
                        if self._canMove(grid, col_change * 2, row_change * 2):
                            valid_move = True
                    else:
                        valid_move = True

        elif move[0] == 'f':
        # check fence placement options
            if self.getRemainingFences() > 0:
                if move[1] in 'hv':
                    if move[2] in 'abcdefghi':
                        if move[3] in '987654321':
                            fence_col = 'abcdefghi'.index(move[2])
                            fence_row = '987654321'.index(move[3])
                            
                            # check if the placement is valid and store move if so
                            if self._canPlaceFence(grid, opponent, move[1], fence_col, fence_row):
                                valid_move = True
        
        if valid_move:                
            new_game = game.duplicate()
            new_winner, new_grid, new_players, new_reward = new_game.execute(move)
            if new_players['player1'].getShortestPath(new_grid) < 0 or new_players['player2'].getShortestPath(new_grid) < 0:
                valid_move = False

        return valid_move
    
    
    def _update_counts(self, col, row):
        """
        Updates the number of times a player has visited a cell.
        """
        self.visited_counts[row, col] += 1
        return
    
    
    def getCellVisits(self, col, row):
        """
        Get the number of times a player has visited a specific cell. 
        """
        return self.visited_counts[row, col]
    

    def getVisitedCounts(self):
        """
        Get the array storing the number of times a player has visited each cell. 
        """
        return self.visited_counts
    

    def getName(self):
        """
        Get the stored name of the player.
        """
        return self.name
    

    def getFences(self):
        """
        Get the set of fences placed by the player. 
        """
        return self.fences
    

    def getCoords(self):
        """
        Get the current coordinates of the player.
        """
        return self.col, self.row
    

    def getColor(self):
        """
        Get the color of the player's pawn and fences.
        """
        return self.color
    

    def getRemainingFences(self):
        """
        Get the number of additional fences the player is allowed to place (assuming limit of 10). 
        """
        return 10 - len(self.fences)
    

    def getShortestPath(self, grid, vfences=None, hfences=None, checkOpponent=True):
        """
        Run A* search to find the player's shortest path to their target row. The heuristic used for A* search 
        is the number of rows from the target.
        """

        # save initial coordinates
        start = (self.col, self.row)

        # initialize priority queue with starting point
        open_heap = []
        heapq.heappush(open_heap, (np.abs(self.target_row - self.row), 0, start))

        # keep track of each visited cell and the distance to the gaol from said cell
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

                ## DOUBLE COL AND ROW CHANGE IF THERE IS A PAWN THERE
                if checkOpponent:
                    if grid._validMove(current_col, current_row, col_change, row_change, vfences, hfences):
                        if grid._isPawn(new_col, new_row):
                            if grid._validMove(new_col, new_row, col_change, row_change, vfences, hfences):
                                new_col = current_col + 2 * col_change
                                new_row = current_row + 2 * row_change
                                new_coords = (new_col, new_row)

                    if new_coords in visited_cells:
                        continue

                if grid._validMove(current_col, current_row, col_change, row_change, vfences, hfences):
                    new_distance = current_distance + 1
                    if new_distance < distances.get(new_distance, float('inf')):
                        distances[new_coords] = new_distance
                        total_estimated_distance = new_distance + np.abs(self.target_row - new_row)
                        heapq.heappush(open_heap, (total_estimated_distance, new_distance, new_coords))

        # return -1 if target row is not accessible by the player
        return -1
    
    
    def _checkWin(self):
        """
        Check whether the player has won by reaching their target goal. 
        """
        return self.row == self.target_row


    def _movePawn(self, grid, col_change, row_change):
        """
        Move the pawn by updating relevant variables. 
        """
        
        # update the pawn location in the grid object
        grid._movePawn(self.col, self.row, col_change, row_change)

        # update coordinates and previous visit counts
        self.col += col_change
        self.row += row_change
        self._update_counts(self.col, self.row)

        return
    

    def _canMove(self, grid, col_change, row_change):
        """
        Check whether the pawn can move by referencing the grid object's fence layout.
        """
        return grid._validMove(self.col, self.row, col_change, row_change)
    
    
    def _placeFence(self, grid, orientation, col, row):
        """
        Place a fence by updating relevant variables. 
        """

        # create a new fence object and add it to the grid
        fence = Fence(orientation, col, row, self.color)
        grid._addFence(fence)

        # add the fence to the player's set of placed fences
        self.fences.add(fence)
        
        return 


    def _canPlaceFence(self, grid, opponent, orientation, col, row):
        """
        Check whether a player can place a fence by checking whether it conflicts with existing fences
        or if it would block either player's path to their target row. 
        """

        # create a fence object for the potential new fence
        temp_fence = Fence(orientation, col, row, self.color)

        # check if it conflicts with the existing fences
        if not grid._validFencePlacement(temp_fence):
            return False
        
        # get horizontal and vertical fence sets if the potential new fence was to be placed
        temp_hfences, temp_vfences = grid._testFencePlacement(temp_fence)

        # check path viability for themself
        if not self.getShortestPath(grid, temp_vfences, temp_hfences) >= 0:
            return False
        
        # check path viability for their opponent
        if not opponent.getShortestPath(grid, temp_vfences, temp_hfences) >= 0:
            return False
        
        return True
    

    def updatePrevMove(self, command):
        """
        Update the prev_move and prev_move_onehot values for the player, given a taken command.
        """

        # store command directly
        self.prev_move = command

        # store one-hot array
        if command in ['pw', 'ps', 'pa', 'pd']:
            self.prev_move_onehot = np.array([1 if command==action else 0 for action in ['pw', 'ps', 'pa', 'pd', '']])
        else:
            self.prev_move_onehot = np.array([0, 0, 0, 0, 1])

        return


    def getPrevMove(self, onehot=False):
        """
        Get the previous move that the player took, either as a one-hot vector or as a string. 
        """
        if onehot:
            return self.prev_move_onehot
        else:
            return self.prev_move
    