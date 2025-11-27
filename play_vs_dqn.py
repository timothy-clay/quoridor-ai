from quoridor import *
from dqn import *
import numpy as np
import torch
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm
from time import sleep


if __name__ == "__main__":

    game = Quoridor(True, print_messages = True, sleep=0.1, gs=9)
    winner, grid, players, reward = game.execute('e')

    state_dim = game.state_dim
    n_actions = game.num_actions

    trained_agent = DQN(state_dim, n_actions)
    trained_agent.load_state_dict(torch.load('quoridor_dqn.pth'))

    while winner is None:

        game._iter_gui()

        winner, grid, players, reward = game.execute('e')
        if winner is not None:
            game._refresh()

            terminate = False
            while not terminate:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                        terminate = True
            break
        
        state, prev_move_onehot = game.getState(grid, players)

        s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        pmo_tensor = torch.tensor(prev_move_onehot, dtype=torch.float32).unsqueeze(0)
        q_values = trained_agent(s_tensor, pmo_tensor)[0].detach().cpu().numpy()

        print(q_values)


        for candidate_id in np.argsort(q_values)[::-1]:
            valid_move = players['active_player'].checkMoveValidity(game, grid, players['inactive_player'], ALL_ACTIONS[candidate_id])
            if valid_move:
                action_idx = candidate_id
                break

        winner, grid, players, reward = game.execute(ALL_ACTIONS[action_idx])

        if winner is not None:
            game._refresh()

            terminate = False
            while not terminate:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                        terminate = True
