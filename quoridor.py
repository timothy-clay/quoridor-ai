import numpy as np
import pygame
import time

class Player:
    def __init__(self, x, y, num_walls=10):
        self.x = x
        self.y = y
        self.walls = num_walls

    def get_position(self):
        return [self.x, self.y]

    def use_wall(self):
        if self.walls > 0:
            self.walls -= 1
            return True
        else:
            return False

    
class Grid:
    def __init__(self, size=9):
        self.size = size
        self.pawns = np.zeros((size, size))

        self.horz_walls = np.zeros((size-1, size))
        self.vert_walls = np.zeros((size, size-1))

    def place_wall(self, pos, orient):

        if orient=='H':
            
            if self.horz_walls[pos[0], pos[1]] == 0 and self.horz_walls[pos[0], pos[1]+1] == 0: 
                temp_walls = self.horz_walls.copy()
                temp_walls[pos[0], pos[1]:pos[1]+1] = 1

                if self.valid_path(temp_walls, self.pawns):
                    self.horz_walls = temp_walls
                    return True
                
            return False
                
        elif orient=='V':
                
            if self.vert_walls[pos[0], pos[1]] == 0 and self.vert_walls[pos[0]+1, pos[1]] == 0: 
                temp_walls = self.vert_walls.copy()
                temp_walls[pos[0]:pos[0]+1, pos[1]] = 1

                if self.valid_path(temp_walls, self.pawns):
                    self.vert_walls = temp_walls
                    return True
                
            return False
        
        return False

    # check whether there's a valid path for both pawns to reach the other side
    def valid_path(self, walls, pawns):
        return True


class QuoridorGame: 
    def __init__(self, grid_size=9):
        self.gridSize = grid_size

        self.grid = Grid(size=self.gridSize)
        self.player_1 = Player(self.gridSize//2, self.gridSize-1)
        self.player_2 = Player(self.gridSize//2, 0)

        self.cellSize = 40
        self.screenSize = self.gridSize * self.cellSize
        self.fps = 60

        self.black = (0, 0, 0)
        self.white = (255, 255, 255)

        self.screen = None
        self.clock = None
        self.sleeptime = 0.1

        pygame.init()
        self.screen = pygame.display.set_mode((self.screenSize, self.screenSize))
        pygame.display.set_caption("Shape Placement Grid")
        self.clock = pygame.time.Clock()

        self._refresh()

    def _drawGrid(self, screen):
        for x in range(0, self.screenSize, self.cellSize):
            for y in range(0, self.screenSize, self.cellSize):
                rect = pygame.Rect(x, y, self.cellSize, self.cellSize)
                pygame.draw.rect(screen, self.black, rect, 1)

    def _refresh(self):
        self.screen.fill(self.white)
        self._drawGrid(self.screen)

        pygame.display.flip()
        self.clock.tick(self.fps)
        time.sleep(self.sleeptime)

    def _loop_GUI(self):

        ## Main Loop for the GUI
        running = True
        while running:
            self.screen.fill(self.white)
            self._drawGrid(self.screen)

            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()

    def _main(self):
        pass

if __name__ == "__main__":
    game = QuoridorGame()
    game._loop_GUI()

