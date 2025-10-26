from quoridor import *

game = Quoridor(GUI=True, print_instructions=False, sleep=1, gs=9)


winner = None
while winner is None:
    move = input()
    winner = game.execute(move)


# allow for copying of game state DONE

# define all possible moves from a given state

# allow playing test turns in copied game state (returns new game state)

# define heuristic function and evaluate game state using said heuristic function

# actually implement minimax
