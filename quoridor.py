import numpy as np
import pygame
import time

from player import *
from grid import *
from fence import *

class Quoridor:
    def __init__(self, GUI=True, render_delay_sec=0.1, gs=9):

        # Constants
        self.gridSize = gs
        self.cellSize = 40
        self.screenSize = self.gridSize * self.cellSize
        self.fps = 60
        self.sleeptime = render_delay_sec

        # Basic color definitions
        self.black = (0, 0, 0)
        self.gray = (241, 241, 241)
        self.white = (255, 255, 255)

        # Color palette for shapes
        self.colors = ['#504136', '#F7C59F']  # Taupe, Peach

        # Mapping of color indices to color names (for debugging purposes)
        self.colorIdxToName = {0: "Taupe", 1: "Peach"}

        # Global variables (now instance attributes)
        self.screen = None
        self.clock = None
        self.grid = np.full((self.gridSize, self.gridSize), -1)
        self.currentShapeIndex = 0
        self.currentColorIndex = 0
        self.shapePos = [0, 0]
        self.placedShapes = []
        self.done = False

        self.grid = Grid(self.gridSize)

        self.player1 = Player(self.gridSize // 2, self.gridSize-1, self.colors[0], self)
        self.player2 = Player(self.gridSize // 2, 0, self.colors[1], self)

        # Initialize the graphical interface (if enabled)
        if GUI:
            pygame.init()
            self.screenSize = self.gridSize * self.cellSize
            self.screen = pygame.display.set_mode((self.screenSize, self.screenSize))
            pygame.display.set_caption("Shape Placement Grid")
            self.clock = pygame.time.Clock()

            self._refresh()

    def getPawnPixels(self, x, y):
        x_pixels = int(x * self.cellSize + self.cellSize / 2)
        y_pixels = int(y * self.cellSize + self.cellSize / 2)
        return x_pixels, y_pixels
    
    def getFencePixels(self, x, y, orientation):
        if orientation == 'h':
            x_pixels = int(x * self.cellSize + self.cellSize / 2)
            y_pixels = int(y * self.cellSize + self.cellSize / 2)
        elif orientation == 'v':
            x_pixels = int(x * self.cellSize + self.cellSize / 2)
            y_pixels = int(y * self.cellSize + self.cellSize / 2)
        return x_pixels, y_pixels

    def _drawGrid(self, screen):
        for x in range(0, self.screenSize, self.cellSize):
            for y in range(0, self.screenSize, self.cellSize):
                rect = pygame.Rect(x, y, self.cellSize, self.cellSize)
                pygame.draw.rect(screen, self.white, rect, 2)

    def _refresh(self):
        self.screen.fill(self.gray)
        self._drawGrid(self.screen)

        # Draw the current state of the grid
        self.player1._drawPawn(self.screen)
        self.player2._drawPawn(self.screen)

        pygame.display.flip()
        self.clock.tick(self.fps)
        time.sleep(self.sleeptime)  

    def _loop_gui(self):
        ## Main Loop for the GUI
        running = True
        mode = None
        active_player = self.player1

        while running:
            self.screen.fill(self.gray)
            self._drawGrid(self.screen)

            self.player1._drawPawn(self.screen)
            self.player2._drawPawn(self.screen)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    # step 1: choose an action
                    if mode is None:
                        if event.key == pygame.K_p:
                            mode = "move_pawn"
                            print("Pawn move mode: press an arrow key to move.")
                        elif event.key == pygame.K_f:
                            mode = "place_fence"
                            print("Fence placement mode: press arrow to choose direction.")
                    
                    # step 2: act based on current mode
                    elif mode == "move_pawn":
                        if event.key == pygame.K_w:
                            if active_player._canMove(0, -1):
                                active_player._movePawn(0, -1)
                                print("Pawn moved up.")
                                active_player = self.player2 if active_player==self.player1 else self.player1
                                mode = None  # reset
                            else:
                                print("Cannot move there.")
                            
                        elif event.key == pygame.K_s:
                            if active_player._canMove(0, 1):
                                active_player._movePawn(0, 1)
                                print("Pawn moved down.")
                                active_player = self.player2 if active_player==self.player1 else self.player1
                                mode = None  # reset
                            else:
                                print("Cannot move there.")

                        elif event.key == pygame.K_a:
                            if active_player._canMove(-1, 0):
                                active_player._movePawn(-1, 0)
                                print("Pawn moved left.")
                                active_player = self.player2 if active_player==self.player1 else self.player1
                                mode = None  # reset
                            else:
                                print("Cannot move there.")

                        elif event.key == pygame.K_d:
                            if active_player._canMove(1, 0):
                                active_player._movePawn(1, 0)
                                print("Pawn moved right.")
                                active_player = self.player2 if active_player==self.player1 else self.player1
                                mode = None  # reset
                            else:
                                print("Cannot move there.")

                        else:
                            print("Cancelled pawn move.")
                            mode = None

                    elif mode == "place_fence":
                        if event.key == pygame.K_h:
                            print("Placed horizontal fence.")
                            mode = None
                        elif event.key == pygame.K_v:
                            print("Placed vertical fence.")
                            mode = None
                        elif event.key == pygame.K_ESCAPE:
                            print("Canceled fence placement.")
                            mode = None

            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()

    def _main(self):
        ## Allows manual control over the environment.
        self._loop_gui()

if __name__ == "__main__":
    # printControls() and main() now encapsulated in the class:
    game = Quoridor(True, render_delay_sec=0.1, gs=9)
    game._main()