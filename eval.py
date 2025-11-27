from quoridor import *
from dqn import DQN
from minmax import minimax
import numpy as np
import torch

def dqn_play(game, players, trained_agent):
    state, prev_move_onehot = game.getState(game.grid, players)

    s_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    pmo_tensor = torch.tensor(prev_move_onehot, dtype=torch.float32).unsqueeze(0)
    q_values = trained_agent(s_tensor, pmo_tensor)[0].detach().cpu().numpy()

    for candidate_id in np.argsort(q_values)[::-1]:
        move = ALL_ACTIONS[candidate_id]
        valid_move = players['active_player'].checkMoveValidity(game, game.grid, players['inactive_player'],
                                                                move)
        if valid_move:
            return move

    return None


def run_single_game(depth=2, trained_agent=None):

    game = Quoridor(GUI=False, print_messages=False, sleep=0, gs=9)
    winner, grid, players, reward = game.execute("e")

    # minimax always play first
    minimax_turn = True

    while winner is None:
        if minimax_turn:
            val, best_move = minimax(
                game,
                winner, grid, players, reward,
                depth, float('-inf'), float('inf')
            )
            winner, grid, players, reward = game.execute(best_move)

            if winner:
                return "minimax"
        else:
            move = dqn_play(game, players, trained_agent)

            if move is None:
                return "minimax"

            winner, grid, players, reward = game.execute(move)

            if winner:
                return "dqn"

        minimax_turn = not minimax_turn



def evaluate_many_games(depth, num_game):

    # dqn
    trained_agent = DQN(len(ALL_ACTIONS))
    trained_agent.load_state_dict(torch.load("quoridor_dqn.pth"))

    minimax_wins = 0
    dqn_wins = 0

    for i in range(num_game):
        print(f"\nGame {i+1}/{num_game}")
        result = run_single_game(
            depth=depth,
            trained_agent=trained_agent
        )

        if result == "minimax":
            minimax_wins += 1
            print("Minimax wins!")
        else:
            dqn_wins += 1
            print("DQN wins!")

    # stats
    print("\nStates")
    print(f"Minimax wins : {minimax_wins}")
    print(f"DQN wins : {dqn_wins}")



if __name__ == "__main__":
    evaluate_many_games(depth=2, num_game=100)
