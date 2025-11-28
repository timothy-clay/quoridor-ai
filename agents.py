from quoridor import *
from dqn import *
from minimax import *
import numpy as np
import torch

class Agent:
    def __init__(self, game, player_id):

        self.game = game
        self.player_id = player_id

        if player_id == 0:
            self.player = self.game.player1
        else:
            self.player = self.game.player2

    def takeTurn(self):
        self.game._iter_gui()
        winner, grid, players, reward = self.game.execute('e')
        return winner, grid, players, reward, ''


class DQNAgent(Agent):
    def __init__(self, game, player_id):
        super().__init__(game, player_id)

        n_actions = game.num_actions

        self.dqn = DQN(n_actions)

        if player_id == 0:
            self.dqn.load_state_dict(torch.load('p1_policy.pth'))
        else:
            self.dqn.load_state_dict(torch.load('p2_policy.pth'))

    def takeTurn(self):
        winner, grid, players, reward = self.game.execute('e')

        state, prev_move_onehot = self.game.getState(grid, players)

        s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        pmo_tensor = torch.tensor(prev_move_onehot, dtype=torch.float32).unsqueeze(0)

        q_values = self.dqn(s_tensor, pmo_tensor)[0].detach().cpu().numpy()

        for candidate_id in np.argsort(q_values)[::-1]:
            valid_move = players['active_player'].checkMoveValidity(self.game, grid, players['inactive_player'], ALL_ACTIONS[candidate_id])
            if valid_move:
                action_idx = candidate_id
                break

        winner, grid, players, reward = self.game.execute(ALL_ACTIONS[action_idx])
        self.game._refresh()

        return winner, grid, players, reward, ALL_ACTIONS[action_idx]


class MinimaxAgent(Agent):
    def __init__(self, game, player_id, depth):
        super().__init__(game, player_id)
        self.depth = depth

    def takeTurn(self):

        winner, grid, players, reward = self.game.execute('e')

        score, best_move = minimax_epsilon_greedy(self.game, winner, grid, players, depth=self.depth, alpha=float('-inf'), beta=float('inf'))
        winner, grid, players, reward = self.game.execute(best_move)
        self.game._refresh()

        return winner, grid, players, reward, best_move


class BaselineAgent(MinimaxAgent):
    def __init__(self, game, player_id):
        super().__init__(game, player_id, depth=1)


if __name__=='__main__':

    game = Quoridor(GUI=True, sleep=0.5)

    p1 = DQNAgent(game, player_id=0)
    p2 = DQNAgent(game, player_id=1)

    agents = (p1, p2)

    winner = None
    current_idx = 0

    while winner is None:
        winner, grid, players, reward = agents[current_idx].takeTurn()
        current_idx = 1 - current_idx