def heuristic(grid, players):
    p1 = players['player1']
    p2 = players['player2']

    p1_path = p1.getShortestPath(grid)
    p2_path = p2.getShortestPath(grid)

    # prioritizes more open moves
    p1_moves = len(p1.getValidTurns(grid, p2))
    p2_moves = len(p2.getValidTurns(grid, p1))

    # prioritizes more remaining fences, push agent to make a move
    p1_f = p1.getRemainingFences()
    p2_f = p2.getRemainingFences()

    return ((p2_path - p1_path) + 0.3 * (p1_moves - p2_moves) + 0.7 * (p1_f - p2_f)
    )


def minimax(game, winner, grid, players, reward, depth, alpha, beta):
    if depth == 0 or winner is not None:
        return (reward if isinstance(reward, (int, float))
                else heuristic(grid, players)), None

    active = players['active_player']
    inactive = players['inactive_player']
    options = active.getValidTurns(grid, inactive)

    # if no legal moves
    if not options:
        return heuristic(grid, players), None

    # Maximizing player 1
    if active == players['player1']:
        best_val = float('-inf')
        best_move = None

        for move in options:
            new_game = game.duplicate()
            new_winner, new_grid, new_players, new_reward = new_game.execute(move)

            val, _ = minimax(
                new_game, new_winner, new_grid, new_players, new_reward,
                depth - 1, alpha, beta
            )

            if val > best_val:
                best_val = val
                best_move = move

            alpha = max(alpha, val)
            if beta <= alpha:
                break

        return best_val, best_move

    # Minimizing player 2
    else:
        best_val = float('inf')
        best_move = None

        for move in options:
            new_game = game.duplicate()
            new_winner, new_grid, new_players, new_reward = new_game.execute(move)

            val, _ = minimax(
                new_game, new_winner, new_grid, new_players, new_reward,
                depth - 1, alpha, beta
            )

            if val < best_val:
                best_val = val
                best_move = move

            beta = min(beta, val)
            if beta <= alpha:
                break

        return best_val, best_move

def get_best_minimax_move(game, depth=3):
    winner, grid, players, reward = game.execute("e")
    eval_score, best_move = minimax(
        game,
        winner,
        grid,
        players,
        reward,
        depth,
        float('-inf'),
        float('inf')
    )

    active = players['active_player']
    active_player = active.getName()

    print(f"{active_player} chooses move: {best_move} (eval={eval_score})")

    return best_move
