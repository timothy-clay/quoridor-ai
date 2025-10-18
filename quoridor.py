import sys

BOARD_SIZE = 9

class Quoridor:
    def __init__(self):
        self.players = {'A': [0, BOARD_SIZE // 2], 'B': [BOARD_SIZE - 1, BOARD_SIZE // 2]}
        self.turn = 'A'
        self.h_walls = set()
        self.v_walls = set()
        self.walls_remaining = {'A': 10, 'B': 10}

    def print_board(self):
        # Use Unicode box characters for better visuals
        print("\n   " + " ".join(str(i) for i in range(BOARD_SIZE)))
        for r in range(BOARD_SIZE):
            # Row of squares and vertical walls
            row_str = f"{r:2d} "
            for c in range(BOARD_SIZE):
                # Square content
                if [r, c] == self.players['A']:
                    row_str += "A"
                elif [r, c] == self.players['B']:
                    row_str += "B"
                else:
                    row_str += "."

                # Draw vertical wall to the right if present
                if (r, c) in self.v_walls or (r - 1, c) in self.v_walls:
                    row_str += "│"  # continuous vertical
                else:
                    row_str += " "
            print(row_str)

            # Draw horizontal walls between this row and the next
            if r < BOARD_SIZE - 1:
                wall_row = "   "
                for c in range(BOARD_SIZE):
                    if (r, c) in self.h_walls:
                        wall_row += "──"
                    else:
                        wall_row += "  "
                print(wall_row)

        print(f"\nPlayer {self.turn}'s turn | Walls left: A={self.walls_remaining['A']}  B={self.walls_remaining['B']}\n")

    def move_pawn(self, direction):
        dr, dc = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}.get(direction, (0, 0))
        r, c = self.players[self.turn]
        nr, nc = r + dr, c + dc

        if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
            print("❌ Move out of bounds.")
            return False
        if [nr, nc] == self.players['A'] or [nr, nc] == self.players['B']:
            print("❌ That square is occupied.")
            return False

        self.players[self.turn] = [nr, nc]

        if (self.turn == 'A' and nr == BOARD_SIZE - 1) or (self.turn == 'B' and nr == 0):
            self.print_board()
            print(f"🎉 Player {self.turn} wins!")
            sys.exit(0)

        self.turn = 'B' if self.turn == 'A' else 'A'
        return True

    def place_wall(self, wall_type, r, c):
        if self.walls_remaining[self.turn] <= 0:
            print("❌ No walls remaining.")
            return False

        if wall_type == 'h':
            if not (0 <= r < BOARD_SIZE - 1 and 0 <= c < BOARD_SIZE - 2):
                print("❌ Invalid horizontal wall position.")
                return False
            if ((r, c) in self.h_walls or (r, c + 1) in self.h_walls):
                print("❌ Overlapping wall.")
                return False
            self.h_walls.add((r, c))
            self.h_walls.add((r, c + 1))

        elif wall_type == 'v':
            if not (0 <= r < BOARD_SIZE - 2 and 0 <= c < BOARD_SIZE - 1):
                print("❌ Invalid vertical wall position.")
                return False
            if ((r, c) in self.v_walls or (r + 1, c) in self.v_walls):
                print("❌ Overlapping wall.")
                return False
            self.v_walls.add((r, c))
            self.v_walls.add((r + 1, c))
        else:
            print("❌ Invalid wall type. Use 'h' or 'v'.")
            return False

        self.walls_remaining[self.turn] -= 1
        self.turn = 'B' if self.turn == 'A' else 'A'
        return True

    def play(self):
        print("Welcome to Quoridor!")
        print("Moves: up/down/left/right | Walls: w h/v r c (e.g. 'w h 2 3')")
        while True:
            self.print_board()
            move = input("Enter move: ").strip().lower().split()

            if len(move) == 1:
                if move[0] in ['up', 'down', 'left', 'right']:
                    self.move_pawn(move[0])
                else:
                    print("❌ Invalid command.")
            elif len(move) == 4 and move[0] == 'w':
                _, wall_type, r, c = move
                try:
                    r, c = int(r), int(c)
                except ValueError:
                    print("❌ Invalid coordinates.")
                    continue
                self.place_wall(wall_type, r, c)
            else:
                print("❌ Invalid input. Try again.")


if __name__ == "__main__":
    game = Quoridor()
    game.play()









# AI-provided very rough draft
# need to make it functional and refactor to be well written