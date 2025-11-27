from quoridor import *
from dqn import *
from minimax import *
import numpy as np
import torch

def terminate_sequence():
    terminate = False
    while not terminate:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                terminate = True


def play_vs_minimax(game, depth=2):

    winner, grid, players, reward = game.execute('e')

    winner = None

    while winner is None:

        game._iter_gui()

        winner, grid, players, reward = game.execute('e')
        game._refresh()

        if winner is not None:
            terminate_sequence()
        
        score, best_move = minimax(game, winner, grid, players, depth=depth, alpha=float('-inf'), beta=float('inf'))
        winner, grid, players, reward = game.execute(best_move)
        game._refresh()

        if winner is not None:
            terminate_sequence()


def play_vs_dqn(game):

    winner, grid, players, reward = game.execute('e')

    n_actions = game.num_actions

    trained_agent = DQN(n_actions)
    trained_agent.load_state_dict(torch.load('quoridor_dqn.pth'))

    while winner is None:

        game._iter_gui()

        winner, grid, players, reward = game.execute('e')
        game._refresh()
        
        if winner is not None:
            terminate_sequence()
        
        state, prev_move_onehot = game.getState(grid, players)

        s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        pmo_tensor = torch.tensor(prev_move_onehot, dtype=torch.float32).unsqueeze(0)
        q_values = trained_agent(s_tensor, pmo_tensor)[0].detach().cpu().numpy()

        for candidate_id in np.argsort(q_values)[::-1]:
            valid_move = players['active_player'].checkMoveValidity(game, grid, players['inactive_player'], ALL_ACTIONS[candidate_id])
            if valid_move:
                action_idx = candidate_id
                break

        winner, grid, players, reward = game.execute(ALL_ACTIONS[action_idx])
        game._refresh()

        if winner is not None:
            terminate_sequence()


def dqn_vs_minimax(game, depth=2):

    winner, grid, players, reward = game.execute('e')

    n_actions = game.num_actions

    trained_agent = DQN(n_actions)
    trained_agent.load_state_dict(torch.load('quoridor_dqn.pth'))

    while winner is None:

        score, best_move = minimax(game, winner, grid, players, depth=depth, alpha=float('-inf'), beta=float('inf'))
        winner, grid, players, reward = game.execute(best_move)
        game._refresh()
        
        if winner is not None:
            terminate_sequence()
        
        state, prev_move_onehot = game.getState(grid, players)

        s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        pmo_tensor = torch.tensor(prev_move_onehot, dtype=torch.float32).unsqueeze(0)
        q_values = trained_agent(s_tensor, pmo_tensor)[0].detach().cpu().numpy()

        for candidate_id in np.argsort(q_values)[::-1]:
            valid_move = players['active_player'].checkMoveValidity(game, grid, players['inactive_player'], ALL_ACTIONS[candidate_id])
            if valid_move:
                action_idx = candidate_id
                break

        winner, grid, players, reward = game.execute(ALL_ACTIONS[action_idx])
        game._refresh()

        if winner is not None:
            terminate_sequence()


if __name__ == "__main__":

    game = Quoridor(True, print_messages = True, sleep=0.1, gs=9)
    dqn_vs_minimax(game)
