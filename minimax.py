from quoridor import *
import numpy as np
import random

def heuristic(players, grid):

    player1 = players['player1']
    player2 = players['player2']

    player1_path = player1.getShortestPath(grid)
    player2_path = player2.getShortestPath(grid)

    player1_win = 10 * int(player1.row == player1.target_row)
    player2_win = 10 * int(player2.row == player2.target_row)

    # avoid oscillations
    player1_prev_visits = 0.1 * player1.getCellVisits(player1.col, player1.row)
    player2_prev_visits = 0.1 * player2.getCellVisits(player2.col, player2.row)

    return player1_win - player1_path - player1_prev_visits - player2_win + player2_path + player2_prev_visits # player 1 wants to maximize, player 2 wants to minimize


def minimax_epsilon_greedy(game, winner, grid, players, depth, alpha, beta, epsilon=0.05):

    if np.random.random() < epsilon:
        
        movement_actions = list(range(4))
        fence_actions = list(range(4, len(ALL_ACTIONS)))

        random.shuffle(movement_actions)
        random.shuffle(fence_actions)

        first_block = movement_actions + [fence_actions[0]]
        random.shuffle(first_block)  

        remaining_fences = fence_actions[1:]
        random.shuffle(remaining_fences)

        candidate_actions = first_block + remaining_fences

        for candidate_action in candidate_actions:
            if players['active_player'].checkMoveValidity(game, grid, players['inactive_player'], ALL_ACTIONS[candidate_action]):
                return 0, ALL_ACTIONS[candidate_action]
    
    else:
        return minimax(game, winner, grid, players, depth, alpha, beta)
        
    return None



def minimax(game, winner, grid, players, depth, alpha, beta):

    if depth==0 or winner is not None:
        return heuristic(players, grid), None
    
    best_move = None
    options = players['active_player'].getValidTurns(grid, players['inactive_player'])

    if players['active_player'] == players['player1']:

        max_heuristic = float('-inf')
        for move in options:
            new_game = game.duplicate()
            new_winner, new_grid, new_players, new_reward = new_game.execute(move)
            eval, _ = minimax(new_game, new_winner, new_grid, new_players, depth - 1, alpha, beta)

            if eval > max_heuristic:
                max_heuristic = eval
                best_move = move

            alpha = max(alpha, eval)

            if beta <= alpha:
                break

        return max_heuristic, best_move

    else:

        min_heuristic = float('inf')
        for move in options:
            new_game = game.duplicate()
            new_winner, new_grid, new_players, new_reward = new_game.execute(move)
            eval, _ = minimax(new_game, new_winner, new_grid, new_players, depth - 1, alpha, beta)

            if eval < min_heuristic:
                min_heuristic = eval
                best_move = move

            beta = min(beta, eval)

            if beta <= alpha:
                break

        return min_heuristic, best_move

if __name__=='__main__':
    game = Quoridor(GUI=True, print_messages=False, sleep=0.1, gs=9)

    winner, grid, players, reward = game.execute('e')

    winner = None
    while winner is None:

        score, best_move = minimax(game, winner, grid, players, depth=2, alpha=float('-inf'), beta=float('inf'))

        winner, grid, players, reward = game.execute(best_move)


# potential improvements:
#    store the player's optimal path and update when necessary (after a fence has been placed)
#    approximate distances to goal in heuristic instead of calculating
#    only check if a move is valid (i.e., doesn't block the opponent) after it's been selected
#    iterative deepening (after previous improvements)