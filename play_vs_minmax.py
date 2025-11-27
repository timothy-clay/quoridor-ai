from quoridor import Quoridor
from minmax import get_best_minimax_move
import pygame

if __name__ == "__main__":

    game = Quoridor(GUI=True, print_messages=True, sleep=0.1, gs=9)

    winner, grid, players, reward = game.execute("e")

    while winner is None:

        print("\nYour turn")
        game._iter_gui()
        game._refresh() # update play in gui

        winner, grid, players, reward = game.execute("e")

        if winner is not None:
            print(f"{winner.getName()} wins!")
            game._refresh()
            terminate = False
            while not terminate:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                        terminate = True
            break

        # minmax
        print("\nMinmax Agent turn")

        best_move = get_best_minimax_move(game, depth=3)
        print(f"Minmax move: {best_move}")

        result = game.execute(best_move)
        game._refresh()

        winner, grid, players, reward = result

        if winner is not None:
            print(f"{winner.getName()} wins!")
            game._refresh()
            terminate = False
            while not terminate:
                for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN or event.type == pygame.QUIT:
                        terminate = True
            break
