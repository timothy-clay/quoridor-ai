from quoridor import *
from dqn import *
from dqn_training import dqn_epsilon_greedy
from minimax import minimax_epsilon_greedy
import numpy as np
import torch

class Agent:
    def __init__(self, game, player_id):
        """
        Create an instance of an agent to play a game of Quoridor.
        """

        self.game = game
        self.player_id = player_id

        if player_id == 0:
            self.player = self.game.player1
        else:
            self.player = self.game.player2

    def takeTurn(self):
        """
        Define default implementation of taking a turn, which assumes a user agent.
        """

        self.game._iter_gui()
        winner, grid, players, reward = self.game.execute('e')
        return winner, grid, players, reward, ''


class DQNAgent(Agent):
    def __init__(self, game, player_id):
        """
        Create an instance of an agent to play a game of Quoridor using DQN logic.
        """

        # call super class constructor 
        super().__init__(game, player_id)

        # create agent
        n_actions = game.num_actions
        self.dqn = DQN(n_actions)
        self.device = torch.device("cpu")

        # load weights according to player 1/2
        if player_id == 0:
            self.dqn.load_state_dict(torch.load('p1_policy.pth'))
        else:
            self.dqn.load_state_dict(torch.load('p2_policy.pth'))

    def takeTurn(self):
        """
        Overwrite implementation of taking a turn.
        """

        # get game state
        winner, grid, players, reward = self.game.execute('e')

        # takes informed actions 95% of the time and pseudo-random actions 5% (avoids deterministic behavior)
        action_idx = dqn_epsilon_greedy(self.dqn, grid, players, self.game, 0.05, self.device)

        # execute turn
        winner, grid, players, reward = self.game.execute(ALL_ACTIONS[action_idx])
        self.game._refresh()

        return winner, grid, players, reward, ALL_ACTIONS[action_idx]


class MinimaxAgent(Agent):
    
    def __init__(self, game, player_id, depth):
        """
        Create an instance of an agent to play a game of Quoridor using Minimax logic.
        """

        # call super class constructor 
        super().__init__(game, player_id)

        # store minimax depth to be used
        self.depth = depth

    def takeTurn(self):
        """
        Overwrite implementation of taking a turn.
        """

        # get game state
        winner, grid, players, reward = self.game.execute('e')

        # takes informed actions 95% of the time and pseudo-random actions 5% (avoids deterministic behavior)
        score, best_move = minimax_epsilon_greedy(self.game, winner, grid, players, depth=self.depth, alpha=float('-inf'), beta=float('inf'))
        
        # execute turn
        winner, grid, players, reward = self.game.execute(best_move)
        self.game._refresh()

        return winner, grid, players, reward, best_move


class BaselineAgent(MinimaxAgent):

    def __init__(self, game, player_id):
        """
        Create an instance of an agent to play a game of Quoridor using baseline logic (Minimax with depth=1).
        """
        super().__init__(game, player_id, depth=1)


if __name__=='__main__':
    pass