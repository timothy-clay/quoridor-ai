from quoridor import *
from dqn import *
from minimax import *
from agents import *
import numpy as np
import torch

def terminate_sequence():
    """
    Defines behavior to wait to quit PyGame until user presses another button. 
    """
    terminate = False
    while not terminate:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                terminate = True


def play_vs_agent(game, player=1, opponent='DQN', depth=2):
    """
    Allows the user to play as a specified player against either a DQN opponent or a minimax opponent of specified depth. 
    """

    # create agent player
    if opponent == 'DQN':
        agent = DQNAgent(game, player_id=3-player)
    elif opponent == 'Minimax':
        agent = MinimaxAgent(game, player_id=3-player, depth=depth)
    else:
        raise ValueError

    # get game state
    winner, grid, players, reward = game.execute('e')

    # declare starting player
    active = 0 if player ==1 else 1

    # loop until game is over
    while winner is None:

        # user turn
        if active == 0:
            game._iter_gui()
            winner, grid, players, reward = game.execute('e')

        # minimax turn
        if active == 1:
            winner, grid, players, reward, taken_action = agent.takeTurn()
        
        # refresh GUI
        game._refresh()

        # wait until user key press to quit PyGame
        if winner is not None:
            terminate_sequence()

        # change turns
        active = 1 - active



if __name__ == "__main__":
    game = Quoridor(True, print_messages = True, sleep=0.1, gs=9)
    play_vs_agent(game, player=2, opponent='DQN')
