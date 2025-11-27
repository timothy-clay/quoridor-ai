from quoridor import *
from dqn import * 
import numpy as np
import torch
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm

TEMPERATURE = 100
CHECKPOINT = 2

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, pmo, a, r, s2, pmo2, d):
        self.buffer.append((s, pmo, a, r, s2, pmo2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, pmo, a, r, s2, pmo2, done = map(np.array, zip(*batch))
        return s, pmo, a, r, s2, pmo2, done

    def __len__(self):
        return len(self.buffer)
    

def epsilon_greedy(dqn, grid, players, game, epsilon, device, verbose=False):

    state, prev_move_onehot = game.getState(grid, players)

    if np.random.random() < epsilon:
        movement_actions = list(range(4))
        fence_actions = list(range(4, len(ALL_ACTIONS)))

        random.shuffle(movement_actions)
        random.shuffle(fence_actions)

        first_block = movement_actions + [fence_actions[0]]
        random.shuffle(first_block)  

        remaining_fences = fence_actions[1:]
        random.shuffle(remaining_fences)

        candidate_actions = first_block + remaining_fences
    
    else:
        s_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        pmo_tensor = torch.tensor(prev_move_onehot, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = dqn(s_tensor, pmo_tensor)[0].detach().cpu().numpy()
        candidate_actions = np.argsort(q_values)[::-1]

        if verbose:
            print(f'pw: {q_values[0]:.2f} | ps: {q_values[1]:.2f} | pa: {q_values[2]:.2f} | pd: {q_values[3]:.2f} | f: {np.max(q_values[4:]):.2f}')

    for candidate_action in candidate_actions:
        if players['active_player'].checkMoveValidity(game, grid, players['inactive_player'], ALL_ACTIONS[candidate_action]):
            return candidate_action
        
    # print('hfences:\n', grid.hfences)
    # print('vfences:\n', grid.vfences)
    # print('pawns:\n', grid.pawns)
    # print('active:', players['active_player'].getCoords())
    # print('inactive:', players['inactive_player'].getCoords())
        
    return None


def calc_loss(policy_net, target_net, states, prev_move_onehots, actions, rewards, next_states, next_prev_move_onehots, dones, gamma, device):

    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    prev_move_onehots_t = torch.tensor(prev_move_onehots, dtype=torch.float32, device=device)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
    next_prev_move_onehots_t = torch.tensor(next_prev_move_onehots, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

    q_values = policy_net(states_t, prev_move_onehots_t)
    q_a = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1) 

    if torch.sum(q_a > 1000) > 0:
        raise ValueError

    with torch.no_grad():
        next_q_policy = policy_net(next_states_t, next_prev_move_onehots_t) 
        next_actions = next_q_policy.argmax(dim=1, keepdim=True)
        next_q_target = target_net(next_states_t, next_prev_move_onehots_t) 
        next_q_target_selected = next_q_target.gather(1, next_actions).squeeze(1) 
        target = rewards_t + gamma * (1.0 - dones_t) * next_q_target_selected

    target = torch.clamp(target, -15.0, 20.0)

    loss = F.mse_loss(q_a, target)
    return loss


def evaluate(game, P1, P2, games=200, device='cpu'):
    wins_P1 = 0
    wins_P2 = 0

    rewards_P1 = []
    rewards_P2 = []

    turns = 0
    
    for _ in tqdm(range(games)):

        game.reset()

        # get current state
        winner, grid, players, reward = game.execute('e')
        state, prev_move_onehot = game.getState(grid, players)

        # define starting player
        if players['active_player'] == players['player1']:
            current_player = 0

        game_length = 0

        reward_P1 = 0
        reward_P2 = 0

        while winner is None and game_length < 250:

            if current_player == 0:
                a = epsilon_greedy(P1, grid, players, game, epsilon=0.05, device=device)
            else:
                a = epsilon_greedy(P2, grid, players, game, epsilon=0.05, device=device)

            try:
                winner, grid, players, reward = game.execute(ALL_ACTIONS[a])

                if current_player == 0:
                    reward_P1 += reward
                else:
                    reward_P2 += reward


                current_player = 1 - current_player

                turns += 1
                game_length += 1
            
            except TypeError:
                break
        
        if winner == players['player1']:
            wins_P1 += 1

        elif winner == players['player2']:
            wins_P2 += 1

        rewards_P1.append(reward_P1)
        rewards_P2.append(reward_P2)
    
    return {
        "P2_winrate": wins_P2 / games,
        "P1_winrate": wins_P1 / games,
        "avg_game_length": turns / games,
        "avg_p2_reward": np.mean(rewards_P2),
        "med_p2_reward": np.median(rewards_P2),
        "avg_p1_reward": np.mean(rewards_P1),
        "med_p1_reward": np.median(rewards_P1)
    }


def visualize_game(P1, P2, device='cpu'):
    wins_P1 = 0
    wins_P2 = 0

    rewards_P1 = []
    rewards_P2 = []

    turns = 0
    
    game = Quoridor(GUI=True, sleep=0.5)

    # get current state
    winner, grid, players, reward = game.execute('e')
    state = game.getState(grid, players)

    # define starting player
    if players['active_player'] == players['player1']:
        current_player = 0

    game_length = 0

    while winner is None and game_length < 250:

        try:
            if current_player == 0:
                #print('P1')
                a = epsilon_greedy(P1, grid, players, game, epsilon=0.05, device=device, verbose=False)
            else:
                #print('P2')
                a = epsilon_greedy(P2, grid, players, game, epsilon=0.05, device=device, verbose=False)

            winner, grid, players, reward = game.execute(ALL_ACTIONS[a])

            current_player = 1 - current_player
            game_length += 1
        
        except TypeError:
            break

    pygame.quit()


def train_dqn(game, episodes=10000, batch_size=64, gamma=0.5, lr_p1=1e-4, lr_p2=1e-4, epsilon_start=1, epsilon_end=0.05, epsilon_decay=0.9995):
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    
    state_dim = game.state_dim
    n_actions = game.num_actions

    # create policy and target networks (target network is a copy of the policy network)
    p1_policy_net = DQN(state_dim, n_actions).to(device)
    p1_target_net = DQN(state_dim, n_actions).to(device)

    try:
        p1_policy_net.load_state_dict(torch.load(f'p1_policy_{CHECKPOINT - 1}.pth'))
    except:
        pass

    p1_target_net.load_state_dict(p1_policy_net.state_dict())
    p1_target_net.eval()

    p2_policy_net = DQN(state_dim, n_actions).to(device)
    p2_target_net = DQN(state_dim, n_actions).to(device)

    try:
        p2_policy_net.load_state_dict(torch.load(f'p2_policy_{CHECKPOINT - 1}.pth'))
    except:
        pass

    p2_target_net.load_state_dict(p2_policy_net.state_dict())
    p2_target_net.eval()

    # optimizer, replay buffer, epsilon definitions
    opt_p1 = optim.Adam(p1_policy_net.parameters(), lr=lr_p1)
    opt_p2 = optim.Adam(p2_policy_net.parameters(), lr=lr_p2)

    rb_p1 = ReplayBuffer()
    rb_p2 = ReplayBuffer()

    epsilon = epsilon_start

    total_steps = 0
    train_steps = 0

    for episode in tqdm(range(1, episodes + 1)):

        try:
            game.reset()

            # get current state
            winner, grid, players, reward = game.execute('e')
            state, prev_move_onehot = game.getState(grid, players)
            done = False

            # define starting player
            if players['active_player'] == players['player1']:
                current_player = 0

            episode_length = 0

            while not done and episode_length < 100:
                
                if current_player == 0:
                    action = epsilon_greedy(p1_policy_net, grid, players, game, epsilon, device)
                else:
                    action = epsilon_greedy(p2_policy_net, grid, players, game, epsilon, device)

                # take action and get next state
                try:
                    winner, grid, players, reward = game.execute(ALL_ACTIONS[action])
        
                except TypeError:
                    done = True
                    continue

                if winner is not None:
                    done = True
                
                next_state, next_prev_move_onehot = game.getState(grid, players)

                if current_player == 0:
                    rb_p1.add(state, prev_move_onehot, action, reward, next_state, next_prev_move_onehot, float(done))
                else:
                    rb_p2.add(state, prev_move_onehot, action, reward, next_state, next_prev_move_onehot, float(done))

                state, prev_move_onehot = next_state, next_prev_move_onehot
                
                episode_length += 1
                total_steps += 1

                if total_steps % 4 == 0:
                    
                    if len(rb_p2) >= 2000:
                        s_b, pmo_b, a_b, r_b, s2_b, pmo2_b, done_b = rb_p2.sample(batch_size)
                        loss_p2 = calc_loss(p2_policy_net, p2_target_net, s_b, pmo_b, a_b, r_b, s2_b, pmo2_b, done_b, gamma, device)
                        opt_p2.zero_grad()
                        loss_p2.backward()

                        torch.nn.utils.clip_grad_norm_(p2_policy_net.parameters(), 1.0)
                        opt_p2.step()
                        train_steps += 1

                    if len(rb_p1) >= 2000:
                        s_b, pmo_b, a_b, r_b, s2_b, pmo2_b, done_b = rb_p1.sample(batch_size)
                        loss_p1 = calc_loss(p1_policy_net, p1_target_net, s_b, pmo_b, a_b, r_b, s2_b, pmo2_b, done_b, gamma, device)
                        opt_p1.zero_grad()
                        loss_p1.backward()

                        torch.nn.utils.clip_grad_norm_(p1_policy_net.parameters(), 1.0)
                        opt_p1.step()
                        train_steps += 1

                    if train_steps > 0 and train_steps % 1000 == 0:
                        p1_target_net.load_state_dict(p1_policy_net.state_dict())
                        p2_target_net.load_state_dict(p2_policy_net.state_dict())

                current_player = 1 - current_player

            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            if episode % 500 == 0:
                eval_results = evaluate(game, p1_policy_net, p2_policy_net, games=100, device=device)
                print(f"P1 Win %: {eval_results['P1_winrate']:.2f} | P2 Win %: {eval_results['P2_winrate']:.2f} | Avg. Game Length: {eval_results['avg_game_length']:.0f}    \nAvg. P1 Reward: {eval_results['avg_p1_reward']:.2f} | Med. P1 Reward: {eval_results['med_p1_reward']:.2f}\nAvg. P2 Reward: {eval_results['avg_p2_reward']:.2f} | Med. P2 Reward: {eval_results['med_p2_reward']:.2f}")
                visualize_game(p1_policy_net, p2_policy_net, device=device)
                torch.save(p1_policy_net.state_dict(), f'p1_policy_{episode}_snapshot.pth')
                torch.save(p2_policy_net.state_dict(), f'p2_policy_{episode}_snapshot.pth')

        except KeyboardInterrupt:
            torch.save(p1_policy_net.state_dict(), f'p1_policy_{CHECKPOINT}.pth')
            torch.save(p2_policy_net.state_dict(), f'p2_policy_{CHECKPOINT}.pth')
            return p2_policy_net
        
    torch.save(p1_policy_net.state_dict(), f'p1_policy_{CHECKPOINT}.pth')
    torch.save(p2_policy_net.state_dict(), f'p2_policy_{CHECKPOINT}.pth')

    return p2_policy_net


def heuristic(players, grid):

    player1 = players['player1']
    player2 = players['player2']

    player1_path = player1.getShortestPath(grid, checkOpponent=False)
    player2_path = player2.getShortestPath(grid, checkOpponent=False)

    player1_win = 10 * int(player1.row == player1.target_row)
    player2_win = 10 * int(player2.row == player2.target_row)

    return player1_win + player2_path - player1_path - player2_win


def train_dqn_heuristic(game, episodes=10000, batch_size=64, gamma=0.9, lr_p2=1e-3, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995):
    
    temp = TEMPERATURE
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    
    state_dim = game.state_dim
    n_actions = game.num_actions

    p2_policy_net = DQN(state_dim, n_actions).to(device)
    p2_target_net = DQN(state_dim, n_actions).to(device)

    try:
        p2_policy_net.load_state_dict(torch.load(f'p2_policy_{CHECKPOINT - 1}.pth'))
    except:
        pass

    p2_target_net.load_state_dict(p2_policy_net.state_dict())
    p2_target_net.eval()

    # optimizer, replay buffer, epsilon definitions
    opt_p2 = optim.Adam(p2_policy_net.parameters(), lr=lr_p2)

    rb_p2 = ReplayBuffer()

    epsilon = epsilon_start

    total_steps = 0
    train_steps = 0

    p1_wins, p2_wins = 0, 0
    p2_rewards = []

    for episode in tqdm(range(1, episodes + 1)):

        try:
            game.reset()

            # get current state
            winner, grid, players, reward = game.execute('e')
            state = game.getState(grid, players)
            done = False

            # define starting player
            if players['active_player'] == players['player1']:
                current_player = 0

            episode_length = 0
            p2_reward = 0

            while not done and episode_length < 250:
                
                if current_player == 0:

                    all_options = ALL_ACTIONS.copy()
                    random.shuffle(all_options)

                    valid_options = []
                    evals = []

                    for move in all_options:
                        if players['active_player'].checkMoveValidity(game, grid, players['inactive_player'], move):
                            new_game = game.duplicate()
                            _, new_grid, new_players, _ = new_game.execute(move)
                            eval = heuristic(new_players, new_grid)

                            valid_options.append(move)
                            evals.append(eval)

                    try:
                        best_moves_idx = np.argsort(evals)[::-1][:min(len(evals), 5)]

                        best_moves = np.array(valid_options)[best_moves_idx]
                        weights = torch.softmax(torch.tensor(np.array(evals)[best_moves_idx], dtype=float) / temp, dim=0)

                        opt_move = random.choices(best_moves, weights=weights)[0]
                        
                        action = ALL_ACTIONS.index(opt_move)
                        
                    except IndexError:
                        done=True
                        continue

                else:
                    action = epsilon_greedy(p2_policy_net, grid, players, game, epsilon, device)

                # take action and get next state
                try:
                    winner, grid, players, reward = game.execute(ALL_ACTIONS[action])
                    heuristic(players, grid)
                    p2_reward += reward
        
                except TypeError:
                    done = True
                    continue

                if winner is not None:
                    if winner == players['player2']:
                        p2_wins += 1
                    else:
                        p1_wins += 1
                    done = True

                next_state = game.getState(grid, players)

                if current_player == 1:
                    rb_p2.add(state, action, reward, next_state, float(done))

                state = next_state
                
                episode_length += 1
                total_steps += 1

                if total_steps % 4 == 0:
                    
                    if len(rb_p2) >= 2000:
                        s_b, a_b, r_b, s2_b, done_b = rb_p2.sample(batch_size)
                        loss_p2 = calc_loss(p2_policy_net, p2_target_net, s_b, a_b, r_b, s2_b, done_b, gamma, device)
                        opt_p2.zero_grad()
                        loss_p2.backward()

                        torch.nn.utils.clip_grad_norm_(p2_policy_net.parameters(), 1.0)
                        opt_p2.step()
                        train_steps += 1

                    if train_steps > 0 and train_steps % 100 == 0:
                        p2_target_net.load_state_dict(p2_policy_net.state_dict())

                current_player = 1 - current_player

            p2_rewards.append(p2_reward)

            if episode % 100 == 0 and episode > 0: 
                print(f'P1 Win %: {(p1_wins/100):.2f} | P2 Win %: {(p2_wins/100):.2f} | Temp: {temp}\nAvg. Reward: {np.mean(p2_rewards):.2f} | Med. Reward: {np.median(p2_rewards):.2f}')

                if p2_wins / (p1_wins + p2_wins) >= 0.6:
                    temp = min(temp / 2, 0.1)

                elif p2_wins / (p1_wins + p2_wins) <= 0.4:
                    temp *= 2

                p1_wins, p2_wins = 0, 0
            
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

        except KeyboardInterrupt:
            torch.save(p2_policy_net.state_dict(), f'p2_policy_{CHECKPOINT}.pth')
            return p2_policy_net
        
    torch.save(p2_policy_net.state_dict(), f'p2_policy_{CHECKPOINT}.pth')

    return p2_policy_net


if __name__=="__main__":
    game = Quoridor(GUI=False)#True, sleep=0.5)   # Replace with your real environment
    trained_net = train_dqn(game)
    torch.save(trained_net.state_dict(), 'quoridor_dqn.pth')