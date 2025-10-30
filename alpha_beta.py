from quoridor import *
import numpy as np

def heuristic(players, grid):

    player1 = players['player1']
    player2 = players['player2']

    player1_path = player1._getShortestPath(grid)
    player2_path = player2._getShortestPath(grid)

    return player2_path - player1_path # player 1 wants to maximize, player 2 wants to minimize

game = Quoridor(GUI=True, print_instructions=False, sleep=1, gs=9)

winner, grid, players = game.execute('e')

winner = None
while winner is None:
    options = players['active_player'].getValidTurns(grid, players['inactive_player'])

    max_heuristic = -100
    for move in options:
        new_game = game.duplicate()
        test_winner, test_grid, test_players = new_game.execute(move)

        test_heuristic = heuristic(test_players, test_grid)

        if test_heuristic > max_heuristic:
            best_move = move
            max_heuristic = test_heuristic

    winner, grid, players = game.execute(best_move)


# allow for copying of game state DONE

# define all possible moves from a given state DONE

# allow playing test turns in copied game state (returns new game state) DONE

# define heuristic function and evaluate game state using said heuristic function

# actually implement minimax
