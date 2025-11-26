from quoridor import *
from alpha_beta import *
import numpy as np

def heuristic(players, grid):

    p1 = players["player1"]
    p2 = players["player2"]

    p1_path = p1.getShortestPath(grid)
    p2_path = p2.getShortestPath(grid)

    # only handle error and reward at the state for faster run time, assisted with chatgpt
    if p1_path == float("inf"): return -9999
    if p2_path == float("inf"): return 9999

    if p1.row == p1.target_row: return 5000
    if p2.row == p2.target_row: return -5000

    # player 1 wants to maximize, player 2 wants to minimize
    # shorter p1_path higher score, shorter p2_path smaller score
    score = p2_path - p1_path

    return score


# minimax
def minimax(game, winner, grid, players, depth, alpha, beta):

    if depth==0 or winner is not None:
        return heuristic(players, grid), None

    best_move = None
    options = players['active_player'].getValidTurns(grid, players['inactive_player'])

    if players['active_player'] == players['player1']:

        max_heuristic = float('-inf')
        for move in options:
            new_game = game.duplicate()
            new_winner, new_grid, new_players = new_game.execute(move)
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
            new_winner, new_grid, new_players = new_game.execute(move)
            eval, _ = minimax(new_game, new_winner, new_grid, new_players, depth - 1, alpha, beta)

            if eval < min_heuristic:
                min_heuristic = eval
                best_move = move

            beta = min(beta, eval)

            if beta <= alpha:
                break

        return min_heuristic, best_move

game = Quoridor(GUI=True, print_messages=False, sleep=0.1, gs=9)

winner, grid, players = game.execute('e')

while winner is None:

    score, best_move = minimax(game, winner, grid, players, depth=5, alpha=float('-inf'), beta=float('inf'))

    print(score, best_move)
    winner, grid, players = game.execute(best_move)

print(f'{winner} wins!')