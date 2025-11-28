from quoridor import *
from dqn import *
from minimax import *
from agents import *
import numpy as np
from tqdm import tqdm
import json


def dqn_vs_minimax(games=100, minimax_depth=3):

    results = {'winners':[],
               'p1s':[],
               'chosen_actions':[],
               'cumulative_rewards':[], 
               'game_lengths':[]}
    
    for _ in tqdm(range(games)):

        game = Quoridor(GUI=False)

        random_idx = np.random.randint(2)

        agents = {}
        agents[random_idx] = DQNAgent(game, player_id=random_idx)
        agents[1 - random_idx] = MinimaxAgent(game, player_id=1-random_idx, depth=minimax_depth)

        current_idx = 0

        dqn_actions = []
        minimax_actions = []

        dqn_reward = 0
        minimax_reward = 0

        winner = None
        game_length = 0
        while winner is None and game_length < 250:
            winner, grid, players, reward, taken_action = agents[current_idx].takeTurn()

            if current_idx == random_idx:
                dqn_actions.append(taken_action)
                dqn_reward += reward
            else:
                minimax_actions.append(taken_action)
                minimax_reward += reward

            current_idx = 1 - current_idx
            game_length += 1

        if random_idx == 0:
            results['p1s'].append('dqn')

            if winner == players['player1']:
                results['winners'].append('dqn')
            elif winner == players['player2']:
                results['winners'].append('minimax')
            else:
                results['winners'].append('none')
        else:
            results['p1s'].append('minimax')
            
            if winner == players['player1']:
                results['winners'].append('minimax')
            elif winner == players['player2']:
                results['winners'].append('dqn')
            else:
                results['winners'].append('none')

        results['chosen_actions'].append({'dqn':dqn_actions, 'minimax':minimax_actions})
        results['cumulative_rewards'].append({'dqn':dqn_reward, 'minimax':minimax_reward})
        results['game_lengths'].append(game_length)

    with open("dqn_vs_minimax.json", "w") as f:
        json.dump(results, f, indent=4)


def minimax_vs_baseline(games=100, minimax_depth=3):

    results = {'winners':[],
               'p1s':[],
               'chosen_actions':[],
               'cumulative_rewards':[], 
               'game_lengths':[]}
    
    for _ in tqdm(range(games)):

        game = Quoridor(GUI=False)

        random_idx = np.random.randint(2)

        agents = {}
        agents[random_idx] = MinimaxAgent(game, player_id=random_idx, depth=minimax_depth)
        agents[1 - random_idx] = BaselineAgent(game, player_id=1-random_idx)

        current_idx = 0

        minimax_actions = []
        baseline_actions = []

        minimax_reward = 0
        baseline_reward = 0

        winner = None
        game_length = 0
        while winner is None and game_length < 250:
            winner, grid, players, reward, taken_action = agents[current_idx].takeTurn()

            if current_idx == random_idx:
                minimax_actions.append(taken_action)
                minimax_reward += reward
            else:
                baseline_actions.append(taken_action)
                baseline_reward += reward

            current_idx = 1 - current_idx
            game_length += 1

        if random_idx == 0:
            results['p1s'].append('minimax')

            if winner == players['player1']:
                results['winners'].append('minimax')
            elif winner == players['player2']:
                results['winners'].append('baseline')
            else:
                results['winners'].append('none')
        else:
            results['p1s'].append('baseline')
            
            if winner == players['player1']:
                results['winners'].append('baseline')
            elif winner == players['player2']:
                results['winners'].append('minimax')
            else:
                results['winners'].append('none')

        results['chosen_actions'].append({'minimax':minimax_actions, 'baseline':baseline_actions})
        results['cumulative_rewards'].append({'minimax':minimax_reward, 'baseline':baseline_reward})
        results['game_lengths'].append(game_length)

    with open("minimax_vs_baseline.json", "w") as f:
        json.dump(results, f, indent=4)


def dqn_vs_baseline(games=100):

    results = {'winners':[],
               'p1s':[],
               'chosen_actions':[],
               'cumulative_rewards':[], 
               'game_lengths':[]}
    
    for _ in tqdm(range(games)):

        game = Quoridor(GUI=False)

        random_idx = np.random.randint(2)

        agents = {}
        agents[random_idx] = DQNAgent(game, player_id=random_idx)
        agents[1 - random_idx] = BaselineAgent(game, player_id=1-random_idx)

        current_idx = 0

        dqn_actions = []
        baseline_actions = []

        dqn_reward = 0
        baseline_reward = 0

        winner = None
        game_length = 0
        while winner is None and game_length < 250:
            winner, grid, players, reward, taken_action = agents[current_idx].takeTurn()

            if current_idx == random_idx:
                dqn_actions.append(taken_action)
                dqn_reward += reward
            else:
                baseline_actions.append(taken_action)
                baseline_reward += reward

            current_idx = 1 - current_idx
            game_length += 1

        if random_idx == 0:
            results['p1s'].append('dqn')

            if winner == players['player1']:
                results['winners'].append('dqn')
            elif winner == players['player2']:
                results['winners'].append('baseline')
            else:
                results['winners'].append('none')
        else:
            results['p1s'].append('baseline')
            
            if winner == players['player1']:
                results['winners'].append('baseline')
            elif winner == players['player2']:
                results['winners'].append('dqn')
            else:
                results['winners'].append('none')

        results['chosen_actions'].append({'dqn':dqn_actions, 'baseline':baseline_actions})
        results['cumulative_rewards'].append({'dqn':dqn_reward, 'baseline':baseline_reward})
        results['game_lengths'].append(game_length)

    with open("dqn_vs_baseline.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__=='__main__':
    dqn_vs_baseline()
    minimax_vs_baseline(minimax_depth=2)
    dqn_vs_minimax(minimax_depth=2)