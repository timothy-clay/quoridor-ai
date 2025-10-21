import numpy as np
import pygame
import time

from player import *
from grid import *
from fence import *

class Quoridor:
    def __init__(self, GUI=True, render_delay_sec=0.1, gs=9):

        # Constants
        self.title_ratio = 1.5
        self.margin = 40
        self.gridSize = gs
        self.cellSize = 40
        self.gridDisplaySide = self.gridSize * self.cellSize
        self.screenSize = self.gridSize * self.cellSize + 2 * self.margin
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

        self.grid._initPawns(self.player1, self.player2)

        # Initialize the graphical interface (if enabled)
        if GUI:
            pygame.init()
            self.screenSize = self.gridSize * self.cellSize + 2 * self.margin
            self.screen = pygame.display.set_mode((self.screenSize, self.screenSize + self.margin * (self.title_ratio - 1)))
            pygame.display.set_caption("Shape Placement Grid")
            self.clock = pygame.time.Clock()

            self._refresh()

    def getPawnPixels(self, x, y):
        x_pixels = int(x * self.cellSize + self.cellSize / 2 + self.margin)
        y_pixels = int(y * self.cellSize + self.cellSize / 2 + self.margin * self.title_ratio)
        return x_pixels, y_pixels
    
    def getFencePixels(self, x, y, orientation):
        if orientation == 'h':
            x_pixels = int(x * self.cellSize + 2 + self.margin)
            y_pixels = int(y * self.cellSize - 2 + self.margin * self.title_ratio)
        elif orientation == 'v':
            x_pixels = int(x * self.cellSize - 2 + self.margin)
            y_pixels = int(y * self.cellSize + 2 + self.margin * self.title_ratio)
        return x_pixels, y_pixels

    def _drawGrid(self, screen):

        font = pygame.font.SysFont('verdana', 30)
        title = font.render('QUORIDOR', True, self.black)
        title_rect = title.get_rect(center=(self.screenSize // 2, self.margin // 2 + 5))
        screen.blit(title, title_rect)

        for i, text in enumerate('ABCDEFGHI'):
            font = pygame.font.SysFont('verdana', 15)

            label_bot = font.render(text, True, (100, 100, 100))
            label_bot_rect = label_bot.get_rect(center=(self.margin + self.cellSize * i + self.cellSize // 2, 
                                                        int(self.margin * self.title_ratio + self.gridDisplaySide + self.margin // 2 - 7)))
            screen.blit(label_bot, label_bot_rect)

            label_side = font.render(str(9-i), True, (100, 100, 100))
            label_side_rect = label_side.get_rect(center=(self.margin // 2 + 7, 
                                                          int(self.cellSize * i + self.cellSize // 2 + self.margin * self.title_ratio)))
            screen.blit(label_side, label_side_rect)

        rect = pygame.Rect(self.margin, int(self.margin * self.title_ratio), self.gridDisplaySide, self.gridDisplaySide)
        pygame.draw.rect(screen, self.gray, rect, self.gridDisplaySide)

        for x in range(self.margin, self.gridDisplaySide+1, self.cellSize):
            for y in range(int(self.margin * self.title_ratio), int(self.gridDisplaySide + self.margin * (self.title_ratio-1) + 1), self.cellSize):
                rect = pygame.Rect(x, y, self.cellSize, self.cellSize)
                pygame.draw.rect(screen, self.white, rect, 2)

    def _refresh(self):
        self.screen.fill(self.white)
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
            self.screen.fill(self.white)
            self._drawGrid(self.screen)

            self.player1._drawPawn(self.screen)
            for fence in self.player1.fences:
                fence._drawFence(self.screen)

            self.player2._drawPawn(self.screen)
            for fence in self.player2.fences:
                fence._drawFence(self.screen)

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
                            if active_player._canMove(self.grid, 0, -1):
                                active_col, active_row = active_player._getCoords()
                                if self.grid._isPawn(active_col, active_row - 1):
                                    if active_player._canMove(self.grid, 0, -2):
                                        active_player._movePawn(self.grid, 0, -2)
                                    else:
                                        continue
                                else:
                                    active_player._movePawn(self.grid, 0, -1)
                                print("Pawn moved up.")
                                active_player = self.player2 if active_player==self.player1 else self.player1
                                mode = None  # reset
                            else:
                                continue
                            
                        elif event.key == pygame.K_s:
                            if active_player._canMove(self.grid, 0, 1):
                                active_col, active_row = active_player._getCoords()
                                if self.grid._isPawn(active_col, active_row + 1):
                                    if active_player._canMove(self.grid, 0, 2):
                                        active_player._movePawn(self.grid, 0, 2)
                                    else:
                                        continue
                                else:
                                    active_player._movePawn(self.grid, 0, 1)
                                print("Pawn moved down.")
                                active_player = self.player2 if active_player==self.player1 else self.player1
                                mode = None  # reset
                            else:
                                continue

                        elif event.key == pygame.K_a:
                            if active_player._canMove(self.grid, -1, 0):
                                active_col, active_row = active_player._getCoords()
                                if self.grid._isPawn(active_col - 1, active_row):
                                    if active_player._canMove(self.grid, -2, 0):
                                        active_player._movePawn(self.grid, -2, 0)
                                    else:
                                        continue
                                else:
                                    active_player._movePawn(self.grid, -1, 0)
                                print("Pawn moved left.")
                                active_player = self.player2 if active_player==self.player1 else self.player1
                                mode = None  # reset
                            else:
                                continue

                        elif event.key == pygame.K_d:
                            if active_player._canMove(self.grid, 1, 0):
                                active_col, active_row = active_player._getCoords()
                                if self.grid._isPawn(active_col + 1, active_row):
                                    if active_player._canMove(self.grid, 2, 0):
                                        active_player._movePawn(self.grid, 2, 0)
                                    else:
                                        continue
                                else:
                                    active_player._movePawn(self.grid, 1, 0)
                                print("Pawn moved right.")
                                active_player = self.player2 if active_player==self.player1 else self.player1
                                mode = None  # reset
                            else:
                                continue

                        else:
                            print("Cancelled pawn move.")
                            mode = None

                    elif mode == "place_fence":
                        if event.key == pygame.K_h:
                            fence_orientation = 'h'
                            mode = "fence_col"
                        elif event.key == pygame.K_v:
                            fence_orientation = 'v'
                            mode = "fence_col"
                        elif event.key == pygame.K_ESCAPE:
                            print("Canceled fence placement.")
                            mode = None

                    elif mode == "fence_col":
                        key_name = pygame.key.name(event.key)
                        if key_name.lower() in 'abcdefghi':
                            fence_col = 'abcdefghi'.index(key_name)
                            mode = "fence_row"
                        else:
                            print("Cancelled fence placement.")
                            mode = None

                    elif mode == "fence_row":
                        key_name = pygame.key.name(event.key)
                        if key_name in '123456789':
                            fence_row = '987654321'.index(key_name)
                            if active_player._canPlaceFence(self.grid, fence_orientation, fence_col, fence_row):
                                active_player._placeFence(self.grid, fence_orientation, fence_col, fence_row)
                            active_player = self.player2 if active_player==self.player1 else self.player1
                            print("Placed fence")
                            mode = None
                        else:
                            print("Cancelled fence placement.")
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