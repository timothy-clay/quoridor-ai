from quoridor import *
from minmax import *


game = Quoridor(GUI=True, print_messages=True)

while True:
    best_move = get_best_minimax_move(game, depth=3)

    winner, grid, players, reward = game.execute(best_move)
    if winner:
        print(f"{winner.getName()} wins!")
        break