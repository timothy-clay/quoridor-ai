from quoridor import *

game = Quoridor(GUI=True, print_instructions=False, sleep=1, gs=9)


# test
winner = None
while winner is None:
    move = input()
    winner = game.execute(move)
