import numpy as np
import pygame
import time

from player import *
from grid import *
from fence import *

class Quoridor:
    def __init__(self, GUI=True, sleep=0.1, gs=9):

        # Constants
        self.title_ratio = 1.5
        self.margin = 40
        self.gridSize = gs
        self.cellSize = 40
        self.gridDisplaySize = self.gridSize * self.cellSize
        self.screenSize = self.gridSize * self.cellSize + 2 * self.margin
        self.fps = 60
        self.sleeptime = sleep

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

        self.player1 = Player('Player 1', self.gridSize // 2, self.gridSize-1, self.colors[0], self)
        self.player2 = Player('Player 2', self.gridSize // 2, 0, self.colors[1], self)

        self.grid._initPawns(self.player1, self.player2)

        self.current_message = self.player1._getName()
        self.current_submessage = "Press 'P' to move pawn, Press 'F' to place a fence"

        # Initialize the graphical interface (if enabled)
        if GUI:
            pygame.init()
            self.screenSize = self.gridSize * self.cellSize + 2 * self.margin
            self.screen = pygame.display.set_mode((self.screenSize, self.screenSize + self.margin * (self.title_ratio - 1) + self.margin))
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
    
    def _printMessage(self, screen, text, subtext, color):
        font = pygame.font.SysFont('verdana', 18)
        message = font.render(text, True, color)
        message_rect = message.get_rect(topleft=(self.margin - 10, self.margin * self.title_ratio + self.gridDisplaySize + self.margin - 12))
        screen.blit(message, message_rect)

        font = pygame.font.SysFont('verdana', 12)
        submessage = font.render(subtext, True, color)
        submessage_rect = message.get_rect(topleft=(self.margin - 10, self.margin * self.title_ratio + self.gridDisplaySize + self.margin + 12))
        screen.blit(submessage, submessage_rect)


    def _drawGrid(self, screen):

        font = pygame.font.SysFont('verdana', 30)
        title = font.render('QUORIDOR', True, self.black)
        title_rect = title.get_rect(center=(self.screenSize // 2, self.margin // 2 + 5))
        screen.blit(title, title_rect)

        for i, text in enumerate('ABCDEFGHI'):
            font = pygame.font.SysFont('verdana', 15)

            label_bot = font.render(text, True, (100, 100, 100))
            label_bot_rect = label_bot.get_rect(center=(self.margin + self.cellSize * i + self.cellSize // 2, 
                                                        int(self.margin * self.title_ratio + self.gridDisplaySize + self.margin // 2 - 7)))
            screen.blit(label_bot, label_bot_rect)

            label_side = font.render(str(9-i), True, (100, 100, 100))
            label_side_rect = label_side.get_rect(center=(self.margin // 2 + 7, 
                                                          int(self.cellSize * i + self.cellSize // 2 + self.margin * self.title_ratio)))
            screen.blit(label_side, label_side_rect)

        rect = pygame.Rect(self.margin, int(self.margin * self.title_ratio), self.gridDisplaySize, self.gridDisplaySize)
        pygame.draw.rect(screen, self.gray, rect, self.gridDisplaySize)

        for x in range(self.margin, self.gridDisplaySize+1, self.cellSize):
            for y in range(int(self.margin * self.title_ratio), int(self.gridDisplaySize + self.margin * (self.title_ratio-1) + 1), self.cellSize):
                rect = pygame.Rect(x, y, self.cellSize, self.cellSize)
                pygame.draw.rect(screen, self.white, rect, 2)

    def _drawPlayerFences(self, screen, player1, player2):

        for i in range(player1._getRemainingFences()):
            rect = pygame.Rect(self.margin + self.gridDisplaySize + 2, int(self.margin * self.title_ratio) + self.gridDisplaySize  - 8 * (i + 1), self.margin - 6, 4)
            pygame.draw.rect(screen, player1._getColor(), rect, 4)

        for i in range(player2._getRemainingFences()):
            rect = pygame.Rect(self.margin + self.gridDisplaySize + 2, int(self.margin * self.title_ratio) + 8 * i + 2, self.margin - 6, 4)
            pygame.draw.rect(screen, player2._getColor(), rect, 4)

    def _refresh(self):
        self.screen.fill(self.white)
        self._drawGrid(self.screen)
        self._drawPlayerFences(self.screen, self.player1, self.player2)
        self._printMessage(self.screen, self.current_message, self.current_submessage, self.black)

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
        inactive_player = self.player2

        while running:
            self.screen.fill(self.white)
            self._drawGrid(self.screen)
            self._drawPlayerFences(self.screen, self.player1, self.player2)
            self._printMessage(self.screen, self.current_message, self.current_submessage, self.black)

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
                            self.current_message = active_player._getName() + ": Move Pawn"
                            self.current_submessage = "Move using W/A/S/D keys."
                        elif event.key == pygame.K_f:
                            mode = "place_fence"
                            self.current_message = active_player._getName() + ": Place Fence"
                            self.current_submessage = "Press 'H' for horizontal fence, Press 'V' for vertical fence."
                    
                    # step 2: act based on current mode
                    elif mode == "move_pawn":
                        if event.key == pygame.K_w:
                            if active_player._canMove(self.grid, 0, -1):
                                active_col, active_row = active_player._getCoords()
                                
                                if self.grid._isPawn(active_col, active_row - 1):
                                    if active_player._canMove(self.grid, 0, -2):
                                        active_player._movePawn(self.grid, 0, -2)
                                    else:
                                        self.current_message = active_player._getName()
                                        self.current_submessage = "Invalid pawn movement. Choose a new action (P/F)."
                                        mode = None
                                else:
                                    active_player._movePawn(self.grid, 0, -1)
                                
                                if active_player._checkWin():
                                    self.current_message = active_player._getName() + ' Wins!'
                                    self.current_submessage = "Press any button to exit."
                                    mode = 'game_over'
                                else:
                                    active_player, inactive_player = inactive_player, active_player
                                    self.current_message = active_player._getName()
                                    self.current_submessage = "Press 'P' to move pawn, Press 'F' to place a fence."
                                    mode = None 

                            else:
                                self.current_message = active_player._getName()
                                self.current_submessage = "Invalid pawn movement. Choose a new action (P/F)."
                                mode = None
                            
                        elif event.key == pygame.K_s:
                            if active_player._canMove(self.grid, 0, 1):
                                active_col, active_row = active_player._getCoords()
                                if self.grid._isPawn(active_col, active_row + 1):
                                    if active_player._canMove(self.grid, 0, 2):
                                        active_player._movePawn(self.grid, 0, 2)
                                    else:
                                        self.current_message = active_player._getName()
                                        self.current_submessage = "Invalid pawn movement. Choose a new action (P/F)."
                                        mode = None
                                else:
                                    active_player._movePawn(self.grid, 0, 1)

                                if active_player._checkWin():
                                    self.current_message = active_player._getName() + ' Wins!'
                                    self.current_submessage = "Press any button to exit."
                                    mode = 'game_over'
                                else:
                                    active_player, inactive_player = inactive_player, active_player
                                    self.current_message = active_player._getName()
                                    self.current_submessage = "Press 'P' to move pawn, Press 'F' to place a fence."
                                    mode = None 
                            else:
                                self.current_message = active_player._getName()
                                self.current_submessage = "Invalid pawn movement. Choose a new action (P/F)."
                                mode = None

                        elif event.key == pygame.K_a:
                            if active_player._canMove(self.grid, -1, 0):
                                active_col, active_row = active_player._getCoords()
                                if self.grid._isPawn(active_col - 1, active_row):
                                    if active_player._canMove(self.grid, -2, 0):
                                        active_player._movePawn(self.grid, -2, 0)
                                    else:
                                        self.current_message = active_player._getName()
                                        self.current_submessage = "Invalid pawn movement. Choose a new action (P/F)."
                                        mode = None
                                else:
                                    active_player._movePawn(self.grid, -1, 0)

                                if active_player._checkWin():
                                    self.current_message = active_player._getName() + ' Wins!'
                                    self.current_submessage = "Press any button to exit."
                                    mode = 'game_over'
                                else:
                                    active_player, inactive_player = inactive_player, active_player
                                    self.current_message = active_player._getName()
                                    self.current_submessage = "Press 'P' to move pawn, Press 'F' to place a fence."
                                    mode = None 
                            else:
                                self.current_message = active_player._getName()
                                self.current_submessage = "Invalid pawn movement. Choose a new action (P/F)."
                                mode = None

                        elif event.key == pygame.K_d:
                            if active_player._canMove(self.grid, 1, 0):
                                active_col, active_row = active_player._getCoords()
                                if self.grid._isPawn(active_col + 1, active_row):
                                    if active_player._canMove(self.grid, 2, 0):
                                        active_player._movePawn(self.grid, 2, 0)
                                    else:
                                        self.current_message = active_player._getName()
                                        self.current_submessage = "Invalid pawn movement. Choose a new action (P/F)."
                                        mode = None
                                else:
                                    active_player._movePawn(self.grid, 1, 0)

                                if active_player._checkWin():
                                    self.current_message = active_player._getName() + ' Wins!'
                                    self.current_submessage = "Press any button to exit."
                                    mode = 'game_over'
                                else:
                                    active_player, inactive_player = inactive_player, active_player
                                    self.current_message = active_player._getName()
                                    self.current_submessage = "Press 'P' to move pawn, Press 'F' to place a fence."
                                    mode = None 
                            else:
                                self.current_message = active_player._getName()
                                self.current_submessage = "Invalid pawn movement. Choose a new action (P/F)."
                                mode = None

                        else:
                            self.current_message = active_player._getName()
                            self.current_submessage = "Invalid pawn movement. Choose a new action (P/F)."
                            mode = None

                    elif mode == "place_fence":
                        if event.key == pygame.K_h:
                            fence_orientation = 'h'
                            mode = "fence_col"
                            self.current_submessage = "Choose a column letter."
                        elif event.key == pygame.K_v:
                            fence_orientation = 'v'
                            mode = "fence_col"
                            self.current_submessage = "Choose a column letter."
                        else:
                            self.current_message = active_player._getName()
                            self.current_submessage = "Invalid fence direction. Choose a new action (P/F)."
                            mode = None

                    elif mode == "fence_col":
                        key_name = pygame.key.name(event.key)
                        if key_name.lower() in 'abcdefghi':
                            fence_col = 'abcdefghi'.index(key_name)
                            mode = "fence_row"
                            self.current_submessage = "Choose a row number."
                        else:
                            self.current_message = active_player._getName()
                            self.current_submessage = "Invalid column letter. Choose a new action (P/F)."
                            mode = None

                    elif mode == "fence_row":
                        key_name = pygame.key.name(event.key)
                        if key_name in '123456789':
                            fence_row = '987654321'.index(key_name)
                            if active_player._getRemainingFences() > 0:
                                if active_player._canPlaceFence(self.grid, inactive_player, fence_orientation, fence_col, fence_row):
                                    active_player._placeFence(self.grid, fence_orientation, fence_col, fence_row)
                                    active_player, inactive_player = inactive_player, active_player
                                    self.current_message = active_player._getName()
                                    self.current_submessage = "Press 'P' to move pawn, Press 'F' to place a fence."
                                    mode = None
                                else:
                                    self.current_message = active_player._getName()
                                    self.current_submessage = "Invalid fence placement. Choose a new action (P/F)."
                            else:
                                self.current_message = active_player._getName()
                                self.current_submessage = "No remaining fences. Choose a new action (P/F)."
                        else:
                            self.current_message = active_player._getName()
                            self.current_submessage = "Invalid row number. Choose a new action (P/F)."

                    elif mode == "game_over":
                        running = False
                        

            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()

    def _main(self):
        self._loop_gui()

if __name__ == "__main__":
    game = Quoridor(True, sleep=0.1, gs=9)
    game._main()