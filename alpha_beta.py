from quoridor import *
import numpy as np

game = Quoridor(GUI=True, print_instructions=False, sleep=0.1, gs=9)

winner, grid, active_player, inactive_player = game.execute('e')

winner = None
while winner is None:
    options = active_player.getValidTurns(grid, inactive_player)
    move = np.random.choice(options)
    winner, grid, active_player, inactive_player = game.execute(move)


# allow for copying of game state DONE

# define all possible moves from a given state

# allow playing test turns in copied game state (returns new game state) 

# define heuristic function and evaluate game state using said heuristic function

# actually implement minimax
