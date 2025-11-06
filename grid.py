import numpy as np

class Grid:
    def __init__(self, gs, hfences=None, vfences=None, pawns=None, fences=None):
        self.gs = gs

        if hfences is not None:
            self.hfences = hfences
        else:
            self.hfences = np.zeros(shape=(gs, gs))

        if vfences is not None:
            self.vfences = vfences
        else:
            self.vfences = np.zeros(shape=(gs, gs))

        if pawns is not None:
            self.pawns = pawns
        else:
            self.pawns = np.zeros(shape=(gs, gs))

        if fences is not None:
            self.fences = fences
        else:
            self.fences = set()

    def duplicate(self):
        return Grid(self.gs, self.hfences.copy(), self.vfences.copy(), self.pawns.copy(), self.fences.copy())


    def _initPawns(self, player1, player2):
        for player in (player1, player2):
            col, row = player.getCoords()
            self.pawns[row, col] = 1

    def _movePawn(self, old_col, old_row, col_change, row_change):
        self.pawns[old_row, old_col] = 0
        self.pawns[old_row + row_change, old_col + col_change] = 1
        return
    
    def _isPawn(self, col, row):
        if self.pawns[row, col] == 1:
            return True
        return False

    def _addFence(self, fence):

        col, row, orientation = fence.getCoords()

        if orientation == 'h':
            self.hfences[row, col:col+2] = 1

        elif orientation == 'v':
            self.vfences[row:row+2, col] = 1

        self.fences.add(fence)

    def _testFencePlacement(self, fence):

        col, row, orientation = fence.getCoords()

        temp_hfences, temp_vfences = self.hfences.copy(), self.vfences.copy()

        if orientation == 'h':
            temp_hfences[row, col:col+2] = 1

        elif orientation == 'v':
            temp_vfences[row:row+2, col] = 1

        return temp_hfences, temp_vfences

    def getPawns(self):
        return self.pawns

    def getHFences(self):
        return self.hfences
    
    def getVFences(self):
        return self.vfences
    
    def _validFencePlacement(self, fence):

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
        if col + col_change >= self.gs or row + row_change >= self.gs:
            return False
        
        elif col + col_change < 0 or row + row_change < 0:
            return False

        if col_change != 0:

            if vfences is None:
                vfences = self.vfences

            # check for vertical fences
            if col_change > 0:

                # check for fence on new cell
                if vfences[row, col + col_change] == 1:
                    return False
                
            elif col_change < 0:
                
                # check for fence on current cell
                if vfences[row, col + col_change + 1] == 1:
                    return False

        elif row_change != 0:

            if hfences is None:
                hfences = self.hfences

            # check for horizontal fences
            if row_change > 0:
                
                # check for fence on new cell
                if hfences[row + row_change, col] == 1:
                    return False

            elif row_change < 0:
                
                # check for fence on current cell
                if hfences[row + row_change + 1, col] == 1:
                    return False
        
        return True

        

    