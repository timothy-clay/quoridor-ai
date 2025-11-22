from quoridor import *
from dqn import * 
import numpy as np
import torch
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm
from time import sleep

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
        
        state = game.getState(grid, players)

        s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q_values = trained_agent(s_tensor)[0].detach().cpu().numpy()


        for candidate_id in np.argsort(q_values):
            valid_move = players['active_player'].checkMoveValidity(grid, players['inactive_player'], ALL_ACTIONS[candidate_id])
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
