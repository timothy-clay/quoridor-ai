import numpy as np
import pygame
from pygame import gfxdraw
import time
import torch

from player import *
from grid import *
from fence import *

DIRECTIONS = {'w':[0, -1], 's':[0, 1], 'a':[-1, 0], 'd':[1, 0]}

ALL_ACTIONS = ['pw','ps','pa','pd','fha2','fha3','fha4','fha5','fha6','fha7','fha8','fha1','fhb2','fhb3','fhb4',
               'fhb5','fhb6','fhb7','fhb8','fhb1','fhc2','fhc3','fhc4','fhc5','fhc6','fhc7','fhc8','fhc1','fhd2',
               'fhd3','fhd4','fhd5','fhd6','fhd7','fhd8','fhd1','fhe2','fhe3','fhe4','fhe5','fhe6','fhe7','fhe8',
               'fhe1','fhf2','fhf3','fhf4','fhf5','fhf6','fhf7','fhf8','fhf1','fhg2','fhg3','fhg4','fhg5','fhg6',
               'fhg7','fhg8','fhg1','fhh2','fhh3','fhh4','fhh5','fhh6','fhh7','fhh8','fhh1','fvb9','fvb2','fvb3',
               'fvb4','fvb5','fvb6','fvb7','fvb8','fvc9','fvc2','fvc3','fvc4','fvc5','fvc6','fvc7','fvc8','fvd9',
               'fvd2','fvd3','fvd4','fvd5','fvd6','fvd7','fvd8','fve9','fve2','fve3','fve4','fve5','fve6','fve7',
               'fve8','fvf9','fvf2','fvf3','fvf4','fvf5','fvf6','fvf7','fvf8','fvg9','fvg2','fvg3','fvg4','fvg5',
               'fvg6','fvg7','fvg8','fvh9','fvh2','fvh3','fvh4','fvh5','fvh6','fvh7','fvh8','fvi9','fvi2','fvi3',
               'fvi4','fvi5','fvi6','fvi7','fvi8']

class Quoridor:
    def __init__(self, GUI=True, print_messages=False, sleep=0.1, gs=9, 
                 grid=None, players=None, active_player=None):
        """
        Create an instance of a Quoridor game.
        """

        # display settings
        self.GUI = GUI
        self.gridSize = gs
        self.title_ratio = 1.5
        self.margin = 40
        self.cellSize = 40
        self.fps = 60
        self.gridDisplaySize = self.gridSize * self.cellSize
        self.screenSize = self.gridSize * self.cellSize + 2 * self.margin
        self.sleeptime = sleep

        # colors
        self.black = (0, 0, 0)
        self.gray = (241, 241, 241)
        self.white = (255, 255, 255)

        # colors for shapes
        self.colors = ['#504136', '#F7C59F'] 

        # global variables
        self.screen = None
        self.clock = None
        self.done = False

        # init grid and player objects (allow copying)
        self.grid = grid if grid else Grid(self.gridSize)
        self.player1 = players[0] if players else Player('Player 1', self.gridSize // 2, self.gridSize-1, self.colors[0], self.gridSize)
        self.player2 = players[1] if players else Player('Player 2', self.gridSize // 2, 0, self.colors[1], self.gridSize)

        # put the players on the grid
        self.grid._initPawns(self.player1, self.player2)

        # keep track of which player's turn it is
        self.active_player = active_player if active_player else self.player1
        self.inactive_player = self.player2 if self.player1 == self.active_player else self.player1

        # what message should be displayed on screen at any time
        self.current_message = self.active_player.getName()
        self.current_submessage = "Press 'P' to move pawn, Press 'F' to place a fence"

        # whether to print messages in GUI and how much extra space to allocate
        self.print_messages = print_messages
        self.message_space = self.margin if self.print_messages else 0

        # game info for DQN purposes
        self.num_actions = 4 + 8**2 * 2
        self.state_dim = 9**2 * 3 + 2 + 2  # horizontal and fence grid and visited cells (9**2 * 3), player 1 location (2), player 2 location (2)

        # create the display if GUI is specified
        if self.GUI:
            pygame.init()
            self.screenSize = self.gridSize * self.cellSize + 2 * self.margin
            self.screen = pygame.display.set_mode((self.screenSize, self.screenSize + self.margin * (self.title_ratio - 1) + self.message_space))
            pygame.display.set_caption("Quoridor")
            self.clock = pygame.time.Clock()
            self._refresh()

    def duplicate(self):
        """
        Creates a copy of the game state. 
        """

        # copy grid and players
        grid = self.grid.duplicate()
        players = (self.player1.duplicate(), self.player2.duplicate())

        # note which player is active
        active_player = players[0] if self.active_player == self.player1 else players[1]
        
        # return new game with specified state (grid, players, active_player)
        return Quoridor(GUI=False, 
                        print_messages=False, 
                        sleep=0,
                        gs=self.gridSize, 
                        grid=grid, 
                        players=players, 
                        active_player=active_player)
    

    def reset(self):
        """
        Reset the game state.
        """
        self.grid = Grid(self.gridSize)
        self.player1 = Player('Player 1', self.gridSize // 2, self.gridSize-1, self.colors[0], self.gridSize)
        self.player2 = Player('Player 2', self.gridSize // 2, 0, self.colors[1], self.gridSize)

        # put the players on the grid
        self.grid._initPawns(self.player1, self.player2)

        # keep track of which player's turn it is
        self.active_player = self.player1
        self.inactive_player = self.player2


    def _changeTurn(self):
        """
        Swap the active player between players 1 and 2.
        """

        # swap active player
        self.active_player, self.inactive_player = self.inactive_player, self.active_player

        # update messages if messages are printed
        if self.print_messages:
            self.current_message = self.active_player.getName()
            self.current_submessage = "Press 'P' to move pawn, Press 'F' to place a fence"

        return
    

    def execute(self, command):
        """
        Interact with the game state using a text command. 

        To move a pawn, the command is "p", followed by one of "w"/"a"/"s"/"d", depending on the desired direction. 

        To place a fence, the command is "f", followed by either "h"/"v" for a horizontal or vertical fence, followed
        by a letter "A"-"I" to indicate the desired column, followed by a number 1-9 to indicate the desired row. 
        Horizontal fences are placed above the selected cell (and extend two cells' lengths to the right), and vertical 
        fences are placed to the left of the selected cell (and extend down two cells' lengths).

        The command "e" returns the current state of the game. 
        """

        active_orig_path = self.active_player.getShortestPath(self.grid)
        inactive_orig_path = self.inactive_player.getShortestPath(self.grid)

        prev_visits = 0
        fence_penalty = 0
        win_reward = 0

        # export current game state
        if command[0].lower() == 'e':

            players = {
                'player1':self.player1, 
                'player2':self.player2, 
                'active_player':self.active_player, 
                'inactive_player': self.inactive_player
            }

            winner = None
            if self.active_player._checkWin():
                winner = self.active_player

            return winner, self.grid, players, 0

        # move pawn
        elif command[0].lower() == 'p':

            # get direction and corresponding row/col changes
            direction = command[1].lower()
            col_change, row_change = DIRECTIONS[direction]

            # check if the player can move in the selected direction
            if self.active_player._canMove(self.grid, col_change, row_change):

                active_col, active_row = self.active_player.getCoords()
                
                # check if the next cell is occupied by the other player
                if self.grid._isPawn(active_col + col_change, active_row + row_change):

                    # move 2 cells in the specified direction if so (and if possible)
                    if self.active_player._canMove(self.grid, col_change * 2, row_change * 2):
                        prev_visits = self.active_player.getCellVisits(active_col + col_change, active_row + row_change)
                        self.active_player._movePawn(self.grid, col_change * 2, row_change * 2)
                
                # otherwise, move only 1 cell in the specified direction
                else:
                    prev_visits = self.active_player.getCellVisits(active_col + col_change, active_row + row_change)
                    self.active_player._movePawn(self.grid, col_change, row_change)

        # place fence
        elif command[0].lower() == 'f':

            # get fence orientation, column, and row from the command
            orientation = command[1].lower()
            fence_col = 'abcdefghi'.index(command[2].lower())
            fence_row = '987654321'.index(command[3].lower())

            # place fence if it doesn't violate any contraints
            if self.active_player.getRemainingFences() > 0:
                if self.active_player._canPlaceFence(self.grid, self.inactive_player, orientation, fence_col, fence_row):
                    self.active_player._placeFence(self.grid, orientation, fence_col, fence_row)
                    fence_penalty = 1

        active_path = self.active_player.getShortestPath(self.grid)
        inactive_path = self.inactive_player.getShortestPath(self.grid)

        # check if there's a winner
        if self.active_player._checkWin():
            winner = self.active_player
            win_reward = 1

        # swap turns if no winner
        else:
            winner = None
            self._changeTurn()

        # refresh GUI
        if self.GUI:
            self._refresh()

        players = {
            'player1':self.player1, 
            'player2':self.player2, 
            'active_player':self.active_player, 
            'inactive_player': self.inactive_player
        }

        reward = win_reward \
            + 0.01 * (active_orig_path - active_path) \
            - 0.01 * (inactive_orig_path - inactive_path) \
            - 0.05 * fence_penalty \
            - 0.01 * prev_visits

        # ideas: penalize fence placements? penalize game length?
        
        # return the state of the board
        return winner, self.grid, players, reward
    

    def getPawnPixels(self, x, y):
        """
        Return the pixel coordinates of a pawn, given its grid coordinates
        """

        # convert grid coordinates to pixels
        x_pixels = int(x * self.cellSize + self.cellSize / 2 + self.margin)
        y_pixels = int(y * self.cellSize + self.cellSize / 2 + self.margin * self.title_ratio)

        return x_pixels, y_pixels
    
    def getFencePixels(self, x, y, orientation):
        """
        Return the pixel coordinates of a fence, given its grid coordinates
        """

        # horizontal fence
        if orientation == 'h':
            x_pixels = int(x * self.cellSize + 2 + self.margin)
            y_pixels = int(y * self.cellSize - 2 + self.margin * self.title_ratio)

        # vertical fence
        elif orientation == 'v':
            x_pixels = int(x * self.cellSize - 2 + self.margin)
            y_pixels = int(y * self.cellSize + 2 + self.margin * self.title_ratio)

        return x_pixels, y_pixels
    
    def getState(self, grid, players):
        hfences = grid.getHFences()
        vfences = grid.getVFences()
        visited_cells = players['active_player'].getVisitedCounts()

        active_col, active_row = players['active_player'].getCoords()
        active_loc = np.zeros((self.gridSize, self.gridSize))
        active_loc[active_row, active_col] = 1

        opp_col, opp_row = players['inactive_player'].getCoords()
        opp_loc = np.zeros((self.gridSize, self.gridSize))
        opp_loc[opp_row, opp_col] = 1

        active_fences_remaining = np.ones((self.gridSize, self.gridSize)) * players['active_player'].getRemainingFences()
        opp_fences_remaining = np.ones((self.gridSize, self.gridSize)) * players['inactive_player'].getRemainingFences()

        # for FFN
        state = np.stack([
            hfences,
            vfences,
            visited_cells,
            active_loc,
            opp_loc,
            active_fences_remaining,
            opp_fences_remaining
        ], axis=0)

        return state

    
    def _printMessage(self, screen, text, subtext, color):
        """
        Print a message (and sub-message) to the screen. 
        """

        # print (blit) the main text to the screen
        font = pygame.font.SysFont('verdana', 18)
        message = font.render(text, True, color)
        message_rect = message.get_rect(topleft=(self.margin - 10, self.margin * self.title_ratio + self.gridDisplaySize + self.margin - 12))
        screen.blit(message, message_rect)

        # print (blit) the sub-text to the screen
        font = pygame.font.SysFont('verdana', 12)
        submessage = font.render(subtext, True, color)
        submessage_rect = message.get_rect(topleft=(self.margin - 10, self.margin * self.title_ratio + self.gridDisplaySize + self.margin + 12))
        screen.blit(submessage, submessage_rect)

        return


    def _drawGrid(self, screen):
        """
        Draw the game board to the screen.
        """

        # draw the QUORIDOR title at the top
        font = pygame.font.SysFont('verdana', 30)
        title = font.render('QUORIDOR', True, self.black)
        title_rect = title.get_rect(center=(self.screenSize // 2, self.margin // 2 + 5))
        screen.blit(title, title_rect)

        # label rows and columns
        for i, text in enumerate('ABCDEFGHI'):
            font = pygame.font.SysFont('verdana', 15)

            # columns
            label_bot = font.render(text, True, (100, 100, 100))
            label_bot_rect = label_bot.get_rect(center=(self.margin + self.cellSize * i + self.cellSize // 2, 
                                                        int(self.margin * self.title_ratio + self.gridDisplaySize + self.margin // 2 - 7)))
            screen.blit(label_bot, label_bot_rect)

            # rows
            label_side = font.render(str(9-i), True, (100, 100, 100))
            label_side_rect = label_side.get_rect(center=(self.margin // 2 + 7, 
                                                          int(self.cellSize * i + self.cellSize // 2 + self.margin * self.title_ratio)))
            screen.blit(label_side, label_side_rect)

        # draw the background
        rect = pygame.Rect(self.margin, int(self.margin * self.title_ratio), self.gridDisplaySize, self.gridDisplaySize)
        pygame.draw.rect(screen, self.gray, rect, self.gridDisplaySize)

        # draw gaps between cells
        for x in range(self.margin, self.gridDisplaySize+1, self.cellSize):
            for y in range(int(self.margin * self.title_ratio), int(self.gridDisplaySize + self.margin * (self.title_ratio-1) + 1), self.cellSize):
                rect = pygame.Rect(x, y, self.cellSize, self.cellSize)
                pygame.draw.rect(screen, self.white, rect, 2)

        return

    def _drawPawn(self, screen, player):
        """
        Draw the position of a player's pawn on the screen.
        """

        # get the player coordinates as pixels
        col, row = player.getCoords()
        x_pixels, y_pixels = self.getPawnPixels(col, row)

        # create and draw a circle to fill the cell
        radius = int(self.cellSize * 0.4)
        gfxdraw.aacircle(screen, x_pixels, y_pixels, radius, player.getColor())
        gfxdraw.filled_circle(screen, x_pixels, y_pixels, radius, player.getColor())

        return
    
    def _drawFence(self, screen, fence):
        """
        Draw the position of a fence on the screen.
        """

        # get the fence coordinates as pixels
        col, row, orientation = fence.getCoords()
        x_pixels, y_pixels = self.getFencePixels(col, row, orientation)

        # draw horizontal fence
        if orientation == 'h':
            rect = pygame.Rect(x_pixels, y_pixels, self.cellSize * 2 - 4, 4)
            pygame.draw.rect(screen, fence.getColor(), rect)

        # draw vertical fence
        elif orientation == 'v':
            rect = pygame.Rect(x_pixels, y_pixels, 4, self.cellSize * 2 - 4)
            pygame.draw.rect(screen, fence.getColor(), rect)

        return

    def _drawPlayerFences(self, screen, player1, player2):
        """
        Draw the number of remaining fences each player has to the right of the board.
        """

        # draw a line for each of player 1's remaining fences
        for i in range(player1.getRemainingFences()):
            rect = pygame.Rect(self.margin + self.gridDisplaySize + 2, int(self.margin * self.title_ratio) + self.gridDisplaySize  - 8 * (i + 1), self.margin - 6, 4)
            pygame.draw.rect(screen, player1.getColor(), rect, 4)

        # draw a line for each of player 2's remaining fences
        for i in range(player2.getRemainingFences()):
            rect = pygame.Rect(self.margin + self.gridDisplaySize + 2, int(self.margin * self.title_ratio) + 8 * i + 2, self.margin - 6, 4)
            pygame.draw.rect(screen, player2.getColor(), rect, 4)

        return

    def _refresh(self):
        """
        Reset the state of the game visually, usually after an update to the game state has been made. 
        """
        
        # re-draw the screen, the grid, and the remaining fences for each player
        self.screen.fill(self.white)
        self._drawGrid(self.screen)
        self._drawPlayerFences(self.screen, self.player1, self.player2)

        # print the current message to the screen if needed
        if self.print_messages:
            self._printMessage(self.screen, self.current_message, self.current_submessage, self.black)

        # draw the pawns for each player
        self._drawPawn(self.screen, self.player1)
        self._drawPawn(self.screen, self.player2)

        # draw each placed fence for player 1
        for fence in self.player1.getFences():
            self._drawFence(self.screen, fence)

        # draw each placed fence for player
        for fence in self.player2.getFences():
            self._drawFence(self.screen, fence)
        
        pygame.display.flip()
        self.clock.tick(self.fps)
        time.sleep(self.sleeptime)  

    def _loop_gui(self):
        """
        Game logic loop for playing the game via the keyboard. 
        """

        # store the mode (e.g., pawn placement, fence placement, etc.) entered by the previous button press
        mode = None

        running = True
        while running:

            # reset display
            self._refresh()

            # get key press
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    
                    # choose pawn placement or fence placement
                    if mode is None:

                        # pawn placement
                        if event.key == pygame.K_p:
                            mode = "move_pawn"
                            self.current_message = self.active_player.getName() + ": Move Pawn"
                            self.current_submessage = "Move using W/A/S/D keys."

                        # fence placement
                        elif event.key == pygame.K_f:
                            mode = "place_fence"
                            self.current_message = self.active_player.getName() + ": Place Fence"
                            self.current_submessage = "Press 'H' for horizontal fence, Press 'V' for vertical fence."
                    
                    # pawn movement logic
                    elif mode == "move_pawn":

                        # only accept valid directions
                        key_name = pygame.key.name(event.key).lower()
                        if key_name in ['w', 's', 'a', 'd']:

                            # get coordinate changes of movement
                            col_change, row_change = DIRECTIONS[key_name]

                            # check if valid move
                            if self.active_player._canMove(self.grid, col_change, row_change):
                                active_col, active_row = self.active_player.getCoords()

                                # check if can/needs to hop other pawn
                                if self.grid._isPawn(active_col + col_change, active_row + row_change):
                                    if self.active_player._canMove(self.grid, col_change * 2, row_change * 2):
                                        self.active_player._movePawn(self.grid, col_change * 2, row_change * 2)
                                        # check winning condition
                                        if self.active_player._checkWin():
                                            self.current_message = self.active_player.getName() + ' Wins!'
                                            self.current_submessage = "Press any button to exit."
                                            mode = 'game_over'

                                        # swap turn and continue
                                        else:
                                            self._changeTurn()
                                            self.current_message = self.active_player.getName()
                                            self.current_submessage = "Press 'P' to move pawn, Press 'F' to place a fence."
                                            mode = None 
                                    else:
                                        self.current_message = self.active_player.getName()
                                        self.current_submessage = "Invalid pawn movement. Choose a new action (P/F)."
                                        mode = None
                                
                                # move one cell
                                else:
                                    self.active_player._movePawn(self.grid, col_change, row_change)
                                        # check winning condition
                                    if self.active_player._checkWin():
                                        self.current_message = self.active_player.getName() + ' Wins!'
                                        self.current_submessage = "Press any button to exit."
                                        mode = 'game_over'

                                    # swap turn and continue
                                    else:
                                        self._changeTurn()
                                        self.current_message = self.active_player.getName()
                                        self.current_submessage = "Press 'P' to move pawn, Press 'F' to place a fence."
                                        mode = None 
                                
                                
                            # print if invalid move
                            else:
                                self.current_message = self.active_player.getName()
                                self.current_submessage = "Invalid pawn movement. Choose a new action (P/F)."
                                mode = None

                        # print if invalid direction
                        else:
                            self.current_message = self.active_player.getName()
                            self.current_submessage = "Invalid pawn movement. Choose a new action (P/F)."
                            mode = None

                    # fence placement logic (checks if valid orientation is provided)
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
                            self.current_message = self.active_player.getName()
                            self.current_submessage = "Invalid fence direction. Choose a new action (P/F)."
                            mode = None

                    # fence placement logic (retrieves column)
                    elif mode == "fence_col":
                        key_name = pygame.key.name(event.key)
                        if key_name.lower() in 'abcdefghi':
                            fence_col = 'abcdefghi'.index(key_name)
                            mode = "fence_row"
                            self.current_submessage = "Choose a row number."
                        else:
                            self.current_message = self.active_player.getName()
                            self.current_submessage = "Invalid column letter. Choose a new action (P/F)."
                            mode = None

                    # fence placement logic (retrieves row)
                    elif mode == "fence_row":
                        key_name = pygame.key.name(event.key)
                        if key_name in '123456789':
                            fence_row = '987654321'.index(key_name)

                            # checks that fence can be placed
                            if self.active_player.getRemainingFences() > 0:
                                if self.active_player._canPlaceFence(self.grid, self.inactive_player, fence_orientation, fence_col, fence_row):
                                    
                                    # actually place fence, then swap turns
                                    self.active_player._placeFence(self.grid, fence_orientation, fence_col, fence_row)
                                    self._changeTurn()
                                    self.current_message = self.active_player.getName()
                                    self.current_submessage = "Press 'P' to move pawn, Press 'F' to place a fence."

                                    mode = None

                                else:
                                    self.current_message = self.active_player.getName()
                                    self.current_submessage = "Invalid fence placement. Choose a new action (P/F)."
                            else:
                                self.current_message = self.active_player.getName()
                                self.current_submessage = "No remaining fences. Choose a new action (P/F)."
                        else:
                            self.current_message = self.active_player.getName()
                            self.current_submessage = "Invalid row number. Choose a new action (P/F)."

                    # end loop if there's a winner
                    elif mode == "game_over":
                        running = False
                        
            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()

    def _iter_gui(self):
        """
        Game logic loop for playing the game via the keyboard. 
        """

        # store the mode (e.g., pawn placement, fence placement, etc.) entered by the previous button press
        mode = None

        running = True
        while running:

            # reset display
            self._refresh()

            # get key press
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    
                    # choose pawn placement or fence placement
                    if mode is None:

                        # pawn placement
                        if event.key == pygame.K_p:
                            mode = "move_pawn"
                            self.current_message = self.active_player.getName() + ": Move Pawn"
                            self.current_submessage = "Move using W/A/S/D keys."

                        # fence placement
                        elif event.key == pygame.K_f:
                            mode = "place_fence"
                            self.current_message = self.active_player.getName() + ": Place Fence"
                            self.current_submessage = "Press 'H' for horizontal fence, Press 'V' for vertical fence."
                    
                    # pawn movement logic
                    elif mode == "move_pawn":

                        # only accept valid directions
                        key_name = pygame.key.name(event.key).lower()
                        if key_name in ['w', 's', 'a', 'd']:

                            # get coordinate changes of movement
                            col_change, row_change = DIRECTIONS[key_name]

                            # check if valid move
                            if self.active_player._canMove(self.grid, col_change, row_change):
                                active_col, active_row = self.active_player.getCoords()

                                # check if can/needs to hop other pawn
                                if self.grid._isPawn(active_col + col_change, active_row + row_change):
                                    if self.active_player._canMove(self.grid, col_change * 2, row_change * 2):
                                        self.active_player._movePawn(self.grid, col_change * 2, row_change * 2)
                                        # check winning condition
                                        if self.active_player._checkWin():
                                            self.current_message = self.active_player.getName() + ' Wins!'
                                            self.current_submessage = "Press any button to exit."
                                            mode = 'game_over'
                                            running = False

                                        # swap turn and continue
                                        else:
                                            self._changeTurn()
                                            self.current_message = self.active_player.getName()
                                            self.current_submessage = "Press 'P' to move pawn, Press 'F' to place a fence."
                                            mode = None 
                                            running = False
                                    else:
                                        self.current_message = self.active_player.getName()
                                        self.current_submessage = "Invalid pawn movement. Choose a new action (P/F)."
                                        mode = None
                                
                                # move one cell
                                else:
                                    self.active_player._movePawn(self.grid, col_change, row_change)
                                        # check winning condition
                                    if self.active_player._checkWin():
                                        self.current_message = self.active_player.getName() + ' Wins!'
                                        self.current_submessage = "Press any button to exit."
                                        mode = 'game_over'
                                        running = False

                                    # swap turn and continue
                                    else:
                                        self._changeTurn()
                                        self.current_message = self.active_player.getName()
                                        self.current_submessage = "Press 'P' to move pawn, Press 'F' to place a fence."
                                        mode = None 
                                        running = False
                                
                                
                            # print if invalid move
                            else:
                                self.current_message = self.active_player.getName()
                                self.current_submessage = "Invalid pawn movement. Choose a new action (P/F)."
                                mode = None

                        # print if invalid direction
                        else:
                            self.current_message = self.active_player.getName()
                            self.current_submessage = "Invalid pawn movement. Choose a new action (P/F)."
                            mode = None

                    # fence placement logic (checks if valid orientation is provided)
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
                            self.current_message = self.active_player.getName()
                            self.current_submessage = "Invalid fence direction. Choose a new action (P/F)."
                            mode = None

                    # fence placement logic (retrieves column)
                    elif mode == "fence_col":
                        key_name = pygame.key.name(event.key)
                        if key_name.lower() in 'abcdefghi':
                            fence_col = 'abcdefghi'.index(key_name)
                            mode = "fence_row"
                            self.current_submessage = "Choose a row number."
                        else:
                            self.current_message = self.active_player.getName()
                            self.current_submessage = "Invalid column letter. Choose a new action (P/F)."
                            mode = None

                    # fence placement logic (retrieves row)
                    elif mode == "fence_row":
                        key_name = pygame.key.name(event.key)
                        if key_name in '123456789':
                            fence_row = '987654321'.index(key_name)

                            # checks that fence can be placed
                            if self.active_player.getRemainingFences() > 0:
                                if self.active_player._canPlaceFence(self.grid, self.inactive_player, fence_orientation, fence_col, fence_row):
                                    
                                    # actually place fence, then swap turns
                                    self.active_player._placeFence(self.grid, fence_orientation, fence_col, fence_row)
                                    self._changeTurn()
                                    self.current_message = self.active_player.getName()
                                    self.current_submessage = "Press 'P' to move pawn, Press 'F' to place a fence."
                                    running = False

                                    mode = None

                                else:
                                    self.current_message = self.active_player.getName()
                                    self.current_submessage = "Invalid fence placement. Choose a new action (P/F)."
                            else:
                                self.current_message = self.active_player.getName()
                                self.current_submessage = "No remaining fences. Choose a new action (P/F)."
                        else:
                            self.current_message = self.active_player.getName()
                            self.current_submessage = "Invalid row number. Choose a new action (P/F)."

                    # end loop if there's a winner
                    elif mode == "game_over":
                        running = False
                        
            pygame.display.flip()
            self.clock.tick(self.fps)

    def _main(self):
        self._loop_gui()

if __name__ == "__main__":
    game = Quoridor(True, print_messages = True, sleep=0.1, gs=9)
    game._main()