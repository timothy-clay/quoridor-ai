from quoridor import *
import numpy as np

def heuristic(players, grid):

    player1 = players['player1']
    player2 = players['player2']

    player1_path = player1._getShortestPath(grid)
    player2_path = player2._getShortestPath(grid)

    player1_win = 10 * int(player1.row == player1.target_row)
    player2_win = 10 * int(player2.row == player2.target_row)

    return player1_win + player2_path - player1_path - player2_win # player 1 wants to maximize, player 2 wants to minimize



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

game = Quoridor(GUI=True, print_instructions=False, sleep=0.1, gs=9)

winner, grid, players = game.execute('e')

winner = None
while winner is None:

    score, best_move = minimax(game, winner, grid, players, depth=2, alpha=float('-inf'), beta=float('inf'))

    print(score, best_move)
    winner, grid, players = game.execute(best_move)


# potential improvements:
#    store the player's optimal path and update when necessary (after a fence has been placed)
#    approximate distances to goal in heuristic instead of calculating
#    only check if a move is valid (i.e., doesn't block the opponent) after it's been selected
#    iterative deepening (after previous improvements)