import pygame
from pygame import gfxdraw

class Player:
    def __init__(self, x, y, color, game):
        self.x = x
        self.y = y

        color = color.lstrip('#')
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)

        self.color = (r, g, b)
        self.game = game

        self.radius = int(self.game.cellSize * 0.4)

    def _drawPawn(self, screen):
        x_pixels, y_pixels = self.game.get_pixels_from_coords(self.x, self.y)
        gfxdraw.aacircle(screen, x_pixels, y_pixels, self.radius, self.color)
        gfxdraw.filled_circle(screen, x_pixels, y_pixels, self.radius, self.color)
        return

    def _movePawn(self, x_change, y_change):
        self.x += x_change
        self.y += y_change
        return self
    
    def _canMove(self, x_change, y_change):
        if self.x + x_change >= self.game.gridSize or self.y + y_change >= self.game.gridSize:
            return False
        elif self.x + x_change < 0 or self.y + y_change < 0:
            return False
        return True