class Fence:
    def __init__(self, orientation, row, col, game):
        self.orientation = orientation
        self.row = row
        self.col = col

        self.game = game

    def _drawFence(self, screen):
        x_pixels, y_pixels = self.game.getFencePixels(self.x, self.y, self.orientation)
        
        return