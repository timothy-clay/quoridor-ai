import numpy as np

class Grid:
    def __init__(self, gs, hfences=None, vfences=None, pawns=None, fences=None):
        """
        Create a new grid object (with certain optional parameters to allow for copying). 
        """
        
        self.gs = gs

        # initialize arrays to store fence and pawn locations and a set of placed fences 
        self.hfences = hfences if hfences is not None else np.zeros(shape=(gs, gs))
        self.vfences = vfences if vfences is not None else np.zeros(shape=(gs, gs))
        self.pawns = pawns if pawns is not None else np.zeros(shape=(gs, gs))
        self.fences = fences if fences is not None else set()


    def duplicate(self):
        """
        Create a new grid object that has the same game state.
        """
        return Grid(gs=self.gs, 
                    hfences=self.hfences.copy(), 
                    vfences=self.vfences.copy(), 
                    pawns=self.pawns.copy(), 
                    fences=self.fences.copy())
    

    def getPawns(self):
        """
        Get the array containing pawn locations.
        """
        return self.pawns
    

    def getHFences(self):
        """
        Get the array containing the locations of the horizontal fences.
        """
        return self.hfences
    
    
    def getVFences(self):
        """
        Get the array containing the locations of the vertical fences. 
        """
        return self.vfences


    def _initPawns(self, player1, player2):
        """
        Place the pawns on the board by adding them to the pawns array.
        """

        # place each pawn given its current coordinates
        for player in (player1, player2):
            col, row = player.getCoords()
            self.pawns[row, col] = 1


    def _movePawn(self, old_col, old_row, col_change, row_change):
        """
        Update the pawns array after a pawn has moved
        """

        # delete pawn from old location
        self.pawns[old_row, old_col] = 0

        # add pawn to new location
        self.pawns[old_row + row_change, old_col + col_change] = 1

        return
    
    
    def _isPawn(self, col, row):
        """
        Check if there is a pawn at a given coordinate pair. 
        """
        return self.pawns[row, col] == 1


    def _addFence(self, fence):
        """
        Add a new fence to the proper fence array.
        """

        # get the fence's location and orientation
        col, row, orientation = fence.getCoords()

        # if horizontal, add a 1 at the specified coordinate and the next column over
        if orientation == 'h':
            self.hfences[row, col:col+2] = 1

        # if vertical, add a 1 at the specified coordinate and the row below
        elif orientation == 'v':
            self.vfences[row:row+2, col] = 1

        # save fence to set of fences
        self.fences.add(fence)


    def _testFencePlacement(self, fence):
        """
        Create and return test horizontal and vertical fence arrays with the added new fence.
        """

        # get fence info and copy the existing fence arrays
        col, row, orientation = fence.getCoords()
        temp_hfences, temp_vfences = self.hfences.copy(), self.vfences.copy()

        # add horizontal fence
        if orientation == 'h':
            temp_hfences[row, col:col+2] = 1

        # add vertical fence
        elif orientation == 'v':
            temp_vfences[row:row+2, col] = 1

        return temp_hfences, temp_vfences

    
    def _validFencePlacement(self, fence):
        """
        Return whether a proposal fence is valid to place (fits in the grid and doesn't intersect with another fence).
        """

        col, row, orientation = fence.getCoords()

        if orientation == 'h':
            # placement restrictions
            if col >= self.gs - 1 or row <= 0:
                return False

            # check that does not overlap with existing fence
            if self.hfences[row, col] == 1 or self.hfences[row, col+1] == 1:
                return False

            # check that col+1, row-1 is not the start of a fence
            for existing_fence in self.fences:
                fence_col, fence_row, fence_orientation = existing_fence.getCoords()
                if fence_orientation == 'v' and fence_col == col + 1 and fence_row == row - 1:
                    return False


        elif orientation == 'v':
            # placement restrictions
            if col <= 0 or row >= self.gs - 1:
                return False

            # check that does not overlap with existing fence
            if self.vfences[row, col] == 1 or self.vfences[row+1, col] == 1:
                return False

            # check that col-1, row+1 is not the start of a fence
            for existing_fence in self.fences:
                fence_col, fence_row, fence_orientation = existing_fence.getCoords()
                if fence_orientation == 'h' and fence_col == col - 1 and fence_row == row + 1:
                    return False

        return True
    
    
    def _validMove(self, col, row, col_change, row_change, vfences = None, hfences = None):
        """
        Return whether or not a pawn can make a certain move or if it's blocked by a fence (or grid border). 
        """

        # grid border restrictions
        if col + col_change >= self.gs or row + row_change >= self.gs or col + col_change < 0 or row + row_change < 0:
            return False

        # check conflicts with vertical fences
        if col_change != 0:

            # load vfences array
            if vfences is None:
                vfences = self.vfences

            # movement down
            if col_change > 0:

                # check for fence on new cell
                if vfences[row, col + col_change] == 1:
                    return False
                
            # movement up
            elif col_change < 0:
                
                # check for fence on current cell
                if vfences[row, col + col_change + 1] == 1:
                    return False

        # check conflicts with horizontal fences
        elif row_change != 0:

            # load hfences array
            if hfences is None:
                hfences = self.hfences

            # movement right
            if row_change > 0:
                
                # check for fence on new cell
                if hfences[row + row_change, col] == 1:
                    return False

            # movement left
            elif row_change < 0:
                
                # check for fence on current cell
                if hfences[row + row_change + 1, col] == 1:
                    return False
        
        return True
    