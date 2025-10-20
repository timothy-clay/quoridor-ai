import pygame

class Fence:
    def __init__(self, orientation, row, col, color, game):
        self.orientation = orientation
        self.row = row
        self.col = col

        self.color = color

        self.game = game

    def _drawFence(self, screen):
        x_pixels, y_pixels = self.game.getFencePixels(self.col, self.row, self.orientation)
        if self.orientation == 'h':
            rect = pygame.Rect(x_pixels, y_pixels, self.game.cellSize * 2 - 4, 4)
            pygame.draw.rect(screen, self.color, rect)
        elif self.orientation == 'v':
            rect = pygame.Rect(x_pixels, y_pixels, 4, self.game.cellSize * 2 - 4)
            pygame.draw.rect(screen, self.color, rect)
        return