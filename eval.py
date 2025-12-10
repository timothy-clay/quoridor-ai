from quoridor import *
from dqn import *
from minimax import *
from agents import *
import numpy as np
from tqdm import tqdm
import json


def dqn_vs_minimax(games=100, minimax_depth=2):
    """
    Simulates some amount of games with a DQN player playing against a minimax player. Tracks performance metrics throughout. 
    """

    # initialize empty dict to store game results
    results = {'winners':[],
               'p1s':[],
               'chosen_actions':[],
               'cumulative_rewards':[], 
               'game_lengths':[]}
    
    for _ in tqdm(range(games)):
        game = Quoridor(GUI=False)

        # randomly assign players 1 and 2
        random_idx = np.random.randint(2)

        # creates DQN and minimax players
        agents = {}
        agents[random_idx] = DQNAgent(game, player_id=random_idx)
        agents[1 - random_idx] = MinimaxAgent(game, player_id=1-random_idx, depth=minimax_depth)

        # initialize actions and cumulative rewards for the game
        dqn_actions, minimax_actions = [], []
        dqn_reward, minimax_reward = 0, 0

        # initialize game info
        current_idx = 0
        winner = None
        game_length = 0
        
        # take turns until someone wins or the game goes 250+ turns
        while winner is None and game_length < 250:
            winner, grid, players, reward, taken_action = agents[current_idx].takeTurn()

            # save actions and rewards
            if current_idx == random_idx:
                dqn_actions.append(taken_action)
                dqn_reward += reward
            else:
                minimax_actions.append(taken_action)
                minimax_reward += reward

            # switch players
            current_idx = 1 - current_idx
            game_length += 1

        # store which player was Player 1 and who eventually won
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

        # add actions, cumulative rewards, and game length to the results dict
        results['chosen_actions'].append({'dqn':dqn_actions, 'minimax':minimax_actions})
        results['cumulative_rewards'].append({'dqn':dqn_reward, 'minimax':minimax_reward})
        results['game_lengths'].append(game_length)

    # export results to a JSON file
    with open("dqn_vs_minimax.json", "w") as f:
        json.dump(results, f, indent=4)


def minimax_vs_baseline(games=100, minimax_depth=2):
    """
    Simulates some amount of games with a minimax player playing against a baseline player. Tracks performance metrics throughout. 
    """

    # initialize empty dict to store game results
    results = {'winners':[],
               'p1s':[],
               'chosen_actions':[],
               'cumulative_rewards':[], 
               'game_lengths':[]}
    
    for _ in tqdm(range(games)):
        game = Quoridor(GUI=False)

        # randomly assign players 1 and 2
        random_idx = np.random.randint(2)

        # creates minimax and baseline players
        agents = {}
        agents[random_idx] = MinimaxAgent(game, player_id=random_idx, depth=minimax_depth)
        agents[1 - random_idx] = BaselineAgent(game, player_id=1-random_idx)

        # initialize actions and cumulative rewards for the game
        minimax_actions, baseline_actions = [], []
        minimax_reward, baseline_reward = 0, 0

        # initialize game info
        current_idx = 0
        winner = None
        game_length = 0

        # take turns until someone wins or the game goes 250+ turns
        while winner is None and game_length < 250:
            winner, grid, players, reward, taken_action = agents[current_idx].takeTurn()

            # save actions and rewards
            if current_idx == random_idx:
                minimax_actions.append(taken_action)
                minimax_reward += reward
            else:
                baseline_actions.append(taken_action)
                baseline_reward += reward

            # switch players
            current_idx = 1 - current_idx
            game_length += 1

        # store which player was Player 1 and who eventually won
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

        # add actions, cumulative rewards, and game length to the results dict
        results['chosen_actions'].append({'minimax':minimax_actions, 'baseline':baseline_actions})
        results['cumulative_rewards'].append({'minimax':minimax_reward, 'baseline':baseline_reward})
        results['game_lengths'].append(game_length)

    # export results to a JSON file
    with open("minimax_vs_baseline.json", "w") as f:
        json.dump(results, f, indent=4)


def dqn_vs_baseline(games=100):
    """
    Simulates some amount of games with a DQN player playing against a baseline player. Tracks performance metrics throughout. 
    """

    # initialize empty dict to store game results
    results = {'winners':[],
               'p1s':[],
               'chosen_actions':[],
               'cumulative_rewards':[], 
               'game_lengths':[]}
    
    for _ in tqdm(range(games)):
        game = Quoridor(GUI=False)

        # randomly assign players 1 and 2
        random_idx = np.random.randint(2)

        # creates DQN and baseline players
        agents = {}
        agents[random_idx] = DQNAgent(game, player_id=random_idx)
        agents[1 - random_idx] = BaselineAgent(game, player_id=1-random_idx)

        # initialize actions and cumulative rewards for the game
        dqn_actions, baseline_actions = [], []
        dqn_reward, baseline_reward = 0, 0

        # initialize game info
        current_idx = 0
        winner = None
        game_length = 0

        # take turns until someone wins or the game goes 250+ turns
        while winner is None and game_length < 250:
            winner, grid, players, reward, taken_action = agents[current_idx].takeTurn()

            # save actions and rewards
            if current_idx == random_idx:
                dqn_actions.append(taken_action)
                dqn_reward += reward
            else:
                baseline_actions.append(taken_action)
                baseline_reward += reward

            # switch players
            current_idx = 1 - current_idx
            game_length += 1

        # store which player was Player 1 and who eventually won
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

        # add actions, cumulative rewards, and game length to the results dict
        results['chosen_actions'].append({'dqn':dqn_actions, 'baseline':baseline_actions})
        results['cumulative_rewards'].append({'dqn':dqn_reward, 'baseline':baseline_reward})
        results['game_lengths'].append(game_length)

    # export results to a JSON file
    with open("dqn_vs_baseline.json", "w") as f:
        json.dump(results, f, indent=4)


def dqn_vs_dqn(games=100):
    """
    Simulates some amount of games with two DQN player playing against each other. Tracks performance metrics throughout. 
    """

    # initialize empty dict to store game results
    results = {'winners':[],
               'chosen_actions':[],
               'cumulative_rewards':[], 
               'game_lengths':[]}
    
    for _ in tqdm(range(games)):
        game = Quoridor(GUI=True, sleep=0.25)

        # creates DQN players
        agents = {}
        agents[0] = DQNAgent(game, player_id=0)
        agents[1] = DQNAgent(game, player_id=1)

        # initialize actions and cumulative rewards for the game
        p1_dqn_actions, p2_dqn_actions = [], []
        p1_dqn_reward, p2_dqn_reward = 0, 0

        # initialize game info
        current_idx = 0
        winner = None
        game_length = 0

        # take turns until someone wins or the game goes 250+ turns
        while winner is None and game_length < 250:
            winner, grid, players, reward, taken_action = agents[current_idx].takeTurn()

            # save actions and rewards
            if current_idx == 0:
                p1_dqn_actions.append(taken_action)
                p1_dqn_reward += reward
            else:
                p2_dqn_actions.append(taken_action)
                p2_dqn_reward += reward

            # switch players
            current_idx = 1 - current_idx
            game_length += 1

        # store which player eventually won
        if winner == players['player1']:
            results['winners'].append('p1')
        elif winner == players['player2']:
            results['winners'].append('p2')
        else:
            results['winners'].append('none')

        # add actions, cumulative rewards, and game length to the results dict
        results['chosen_actions'].append({'p1':p1_dqn_actions, 'p2':p2_dqn_actions})
        results['cumulative_rewards'].append({'p1':p1_dqn_reward, 'p2':p2_dqn_reward})
        results['game_lengths'].append(game_length)

    # export results to a JSON file
    with open("dqn_vs_dqn.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__=='__main__':
    # dqn_vs_baseline()
    # minimax_vs_baseline()
    # dqn_vs_minimax()
    pass