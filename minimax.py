from quoridor import *
import numpy as np
import random

def heuristic(players, grid):
    """
    Returns a heuristic score given the current state of the game (players and grid).
    """

    # get players 1 and 2 from players dict
    p1 = players['player1']
    p2 = players['player2']

    # measure each player's path to the goal
    p1_path = p1.getShortestPath(grid)
    p2_path = p2.getShortestPath(grid)

    # check whether either play has won
    p1_win = 10 * int(p1.row == p1.target_row)
    p2_win = 10 * int(p2.row == p2.target_row)

    # get the number of fences each player has remaining
    p1_fences = 0.5 * p1.getRemainingFences()
    p2_fences = 0.5 * p2.getRemainingFences()

    # get the number of times each player has visited their current cell
    p1_prev_visits = 0.5 * p1.getCellVisits(p1.col, p1.row)
    p2_prev_visits = 0.5 * p2.getCellVisits(p2.col, p2.row)

    # compute individual scores
    p1_score = p1_win - p1_path - p1_prev_visits + p1_fences
    p2_score = p2_win - p2_path - p2_prev_visits + p2_fences

    # return difference in scores (player 1 wants to maximize, player 2 wants to minimize)
    return p1_score - p2_score


def minimax_epsilon_greedy(game, winner, grid, players, depth, alpha, beta, epsilon=0.05):
    """
    Take an action with a Minimax agent according to a pseudo-epsilon-greedy approach. 
    """

    # take random move with probability epsilon
    if np.random.random() < epsilon:

        # get a list of all movement and fence indices
        movement_actions = list(range(4))
        fence_actions = list(range(4, len(ALL_ACTIONS)))

        # shuffle the fence indices
        random.shuffle(fence_actions)

        # combine movement options and first four fence options, then shuffle those options
        first_block = movement_actions + fence_actions[:4]
        random.shuffle(first_block)  

        # get the remaining fence indices that do not appear in the first block
        remaining_fences = fence_actions[4:]

        # combine the first block of candidate actions with the remaining fences
        candidate_actions = first_block + remaining_fences

        # loop through each candidate action, check if it's valid, and return its index if so
        for candidate_action in candidate_actions:
            if players['active_player'].checkMoveValidity(game, grid, players['inactive_player'], ALL_ACTIONS[candidate_action]):
                return 0, ALL_ACTIONS[candidate_action]
    
    # take action according to minimax policy
    else:
        return minimax(game, winner, grid, players, depth, alpha, beta)
        
    return None



def minimax(game, winner, grid, players, depth, alpha, beta):
    """
    Perform minimax on a certain depth to determine the most optimal move for a player. 
    """

    # return heuristic if terminal recursive state reached
    if depth==0 or winner is not None:
        return heuristic(players, grid), None
    
    best_move = None
    
    # get all valid moves for the player
    options = players['active_player'].getValidTurns(grid, players['inactive_player'])

    # maximizes if player 1 is current player
    if players['active_player'] == players['player1']:

        max_heuristic = float('-inf')

        # evaluate each viable move
        for move in options:
            new_game = game.duplicate()
            new_winner, new_grid, new_players, new_reward = new_game.execute(move)
            eval, _ = minimax(new_game, new_winner, new_grid, new_players, depth - 1, alpha, beta)

            # store best score as found
            if eval > max_heuristic:
                max_heuristic = eval
                best_move = move

            # alpha-beta pruning
            alpha = max(alpha, eval)
            if beta <= alpha:
                break

        # return optimal move and the heuristic that corresponds to it
        return max_heuristic, best_move

    # minimizes if player 2 is current player
    else:

        min_heuristic = float('inf')

        # evaluate each viable move
        for move in options:
            new_game = game.duplicate()
            new_winner, new_grid, new_players, new_reward = new_game.execute(move)
            eval, _ = minimax(new_game, new_winner, new_grid, new_players, depth - 1, alpha, beta)

            # store best score as found
            if eval < min_heuristic:
                min_heuristic = eval
                best_move = move

            # alpha-beta pruning
            beta = min(beta, eval)
            if beta <= alpha:
                break

        # return optimal move and the heuristic that corresponds to it
        return min_heuristic, best_move


if __name__=='__main__':
    pass