from quoridor import *
from dqn import * 
import numpy as np
import torch
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm


class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = map(np.array, zip(*batch))
        return s, a, r, s2, done

    def __len__(self):
        return len(self.buffer)
    

def epsilon_greedy(dqn, grid, players, game, epsilon, device):

    state = game.getState(grid, players)

    if np.random.random() < epsilon:
        movement_actions = list(range(4))
        fence_actions = list(range(4, len(ALL_ACTIONS)))

        random.shuffle(movement_actions)
        random.shuffle(fence_actions)

        first_block = movement_actions + fence_actions[:4]
        random.shuffle(first_block)  

        remaining_fences = fence_actions[4:]
        random.shuffle(remaining_fences)

        candidate_actions = first_block + remaining_fences
    
    else:
        s_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = dqn(s_tensor)[0].detach().cpu().numpy()
        candidate_actions = np.argsort(q_values)[::-1]

    for candidate_action in candidate_actions:
        if players['active_player'].checkMoveValidity(game, grid, players['inactive_player'], ALL_ACTIONS[candidate_action]):
            return candidate_action
        
    # print('hfences:\n', grid.hfences)
    # print('vfences:\n', grid.vfences)
    # print('pawns:\n', grid.pawns)
    # print('active:', players['active_player'].getCoords())
    # print('inactive:', players['inactive_player'].getCoords())
        
    return None


def calc_loss(policy_net, target_net, states, actions, rewards, next_states, dones, gamma, device):

    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

    q_values = policy_net(states_t)
    q_a = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1) 

    with torch.no_grad():
        next_q_policy = policy_net(next_states_t) 
        next_actions = next_q_policy.argmax(dim=1, keepdim=True)
        next_q_target = target_net(next_states_t) 
        next_q_target_selected = next_q_target.gather(1, next_actions).squeeze(1) 
        target = rewards_t + gamma * (1.0 - dones_t) * next_q_target_selected

    loss = F.mse_loss(q_a, target)
    return loss


def evaluate(game, P1, P2, games=200):
    wins_P1 = 0
    wins_P2 = 0

    turns = 0
    
    for _ in range(games):

        game.reset()

        # get current state
        winner, grid, players, reward = game.execute('e')
        state = game.getState(grid, players)

        # define starting player
        if players['active_player'] == players['player1']:
            current_player = 0

        game_length = 0

        while winner is None and game_length < 1000:

            if current_player == 0:
                a = epsilon_greedy(P1, grid, players, game, epsilon=0.05)
            else:
                a = epsilon_greedy(P2, grid, players, game, epsilon=0.05)

            winner, grid, players, reward = game.execute(ALL_ACTIONS[a])

            current_player = 1 - current_player

            turns += 1
            game_length += 1
        
        if winner == players['player1']:
            wins_P1 += 1
        elif winner == players['player2']:
            wins_P2 += 1
    
    return {
        "P2_winrate": wins_P2 / games,
        "P1_winrate": wins_P1 / games,
        "avg_game_length": turns / games
    }


def train_dqn(game, episodes=10000, batch_size=64, gamma=0.99, lr_p1=1e-4, lr_p2=1e-4, epsilon_start=1.0, epsilon_end=0.05, 
              epsilon_decay=0.9995, target_update_interval=1):
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)
    
    state_dim = game.state_dim
    n_actions = game.num_actions

    # create policy and target networks (target network is a copy of the policy network)
    p1_policy_net = DQN(state_dim, n_actions).to(device)
    p1_target_net = DQN(state_dim, n_actions).to(device)
    p1_target_net.load_state_dict(p1_policy_net.state_dict())
    p1_target_net.eval()

    p2_policy_net = DQN(state_dim, n_actions).to(device)
    p2_target_net = DQN(state_dim, n_actions).to(device)
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
    p1_train_counter = 0

    ### CAN ADD SOME CODE HERE TO EVALUATE THE MODEL PERFORMANCES ###

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

            while not done and episode_length < 250:
                
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

                done = winner is not None
                next_state = game.getState(grid, players)

                if current_player == 0:
                    rb_p1.add(state, action, reward, next_state, float(done))
                else:
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

                    if len(rb_p1) >= 2000:
                        p1_train_counter += 1
                        if (p1_train_counter % 5) == 0:
                            s_b, a_b, r_b, s2_b, done_b = rb_p1.sample(batch_size)
                            loss_p1 = calc_loss(p1_policy_net, p1_target_net, s_b, a_b, r_b, s2_b, done_b, gamma, device)
                            opt_p1.zero_grad()
                            loss_p1.backward()

                            torch.nn.utils.clip_grad_norm_(p1_policy_net.parameters(), 1.0)
                            opt_p1.step()
                            train_steps += 1

                    if train_steps > 0 and train_steps % 1000 == 0:
                        p1_target_net.load_state_dict(p1_policy_net.state_dict())
                        p2_target_net.load_state_dict(p2_policy_net.state_dict())

                current_player = 1 - current_player

            # if episode % 1000 == 0:
            #     metrics = evaluate(game, p1_target_net, p2_target_net)
            #     print(f'== Eval @ ep {episode} | P1: {metrics["P1_winrate"]:.3f} | P2: {metrics["P2_winrate"]:.3f} | Game Length: {metrics["avg_game_length"]:.1f}')

            epsilon = max(epsilon_end, epsilon * epsilon_decay)

        except KeyboardInterrupt:
            return p2_policy_net

    return p2_policy_net


if __name__=="__main__":
    game = Quoridor(GUI=False)#True, sleep=0.1)   # Replace with your real environment
    trained_net = train_dqn(game)
    torch.save(trained_net.state_dict(), 'quoridor_dqn.pth')