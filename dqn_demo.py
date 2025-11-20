from quoridor import *
from dqn import * 
import numpy as np
import torch
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm

ALL_ACTIONS = ['pw','ps','pa','pd','fha2','fha3','fha4','fha5','fha6','fha7','fha8','fha1','fhb2','fhb3','fhb4',
               'fhb5','fhb6','fhb7','fhb8','fhb1','fhc2','fhc3','fhc4','fhc5','fhc6','fhc7','fhc8','fhc1','fhd2',
               'fhd3','fhd4','fhd5','fhd6','fhd7','fhd8','fhd1','fhe2','fhe3','fhe4','fhe5','fhe6','fhe7','fhe8',
               'fhe1','fhf2','fhf3','fhf4','fhf5','fhf6','fhf7','fhf8','fhf1','fhg2','fhg3','fhg4','fhg5','fhg6',
               'fhg7','fhg8','fhg1','fhh2','fhh3','fhh4','fhh5','fhh6','fhh7','fhh8','fhh1','fvb9','fvb2','fvb3',
               'fvb4','fvb5','fvb6','fvb7','fvb8','fvc9','fvc2','fvc3','fvc4','fvc5','fvc6','fvc7','fvc8','fvd9',
               'fvd2','fvd3','fvd4','fvd5','fvd6','fvd7','fvd8','fve9','fve2','fve3','fve4','fve5','fve6','fve7',
               'fve8','fvf9','fvf2','fvf3','fvf4','fvf5','fvf6','fvf7','fvf8','fvg9','fvg2','fvg3','fvg4','fvg5',
               'fvg6','fvg7','fvg8','fvh9','fvh2','fvh3','fvh4','fvh5','fvh6','fvh7','fvh8','fvi9','fvi2','fvi3',
               'fvi4','fvi5','fvi6','fvi7','fvi8']

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, s, a, r, s2, d):
        self.buffer.append((s, a, r, s2, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (np.array(s),
                np.array(a),
                np.array(r, dtype=np.float32),
                np.array(s2),
                np.array(d, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)

def train_dqn(game, episodes=10000, batch_size=64, gamma=0.99, lr=1e-3, epsilon_start=1.0, epsilon_end=0.05, 
              epsilon_decay=0.9995, target_update_interval=200):
    
    # what the input for the DQN is going to be
        # horizontal fences, vertical fences, self location, opponent location
    state_dim = game.state_dim

    # the number of total valid actions
    n_actions = game.num_actions

    # create policy and target networks (target network is a copy of the policy network)
    policy_net = DQN(state_dim, n_actions)
    target_net = DQN(state_dim, n_actions)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # optimizer, replay buffer, epsilon definitions
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    buffer = ReplayBuffer()
    epsilon = epsilon_start

    # training loop
    for episode in tqdm(range(episodes)):

        game.reset()

        # get current state
        winner, grid, players = game.execute('e')
        state = game.getState(grid, players)
        done = False

        # define starting player
        if players['active_player'] == players['player1']:
            current_player = 1

        episode_length = 0

        # loop until game ends
        while not done and episode_length < 100:
            
            # get all legal moves for the active player
            #candidate_moves = players['active_player'].getValidTurns(grid, players['inactive_player'])
            #candidate_ids = [ALL_ACTIONS.index(move) for move in candidate_moves]

            # epsilon-greedy
            if np.random.random() < epsilon:

                shuffled_actions = random.sample(ALL_ACTIONS, len(ALL_ACTIONS))

                for candidate_action in shuffled_actions:
                    valid_move = players['active_player'].checkMoveValidity(grid, players['inactive_player'], candidate_action)
                    if valid_move:
                        action_idx = ALL_ACTIONS.index(candidate_action)
                        break
                    
            
            else:

                # get predicted q values using state tensor
                s_tensor = torch.tensor(state).float().unsqueeze(0)
                q_values = policy_net(s_tensor)[0].detach().numpy()

                for candidate_id in np.argsort(q_values):
                    valid_move = players['active_player'].checkMoveValidity(grid, players['inactive_player'], ALL_ACTIONS[action_idx])
                    if valid_move:
                        action_idx = candidate_id
                        break

            # take action and get next state
            winner, grid, players, reward = game.execute(ALL_ACTIONS[action_idx])

            done = winner is not None
            next_state = game.getState(grid, players)

            # adjust reward for the player playing
            corrected_reward = reward if current_player == 1 else -reward

            # add state-action-reward-next_state-winner to replay buffer
            buffer.add(state, action_idx, corrected_reward, next_state, done)      # can state and next state be tensors??

            # update state and swap players
            state = next_state
            current_player = 3 - current_player 

            # update model
            if len(buffer) >= batch_size:

                # sample from the replay buffer
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                # convert to tensors
                states_t = torch.tensor(states, dtype=torch.float32)
                actions_t = torch.tensor(actions, dtype=torch.long)
                rewards_t = torch.tensor(rewards, dtype=torch.float32)
                next_states_t = torch.tensor(next_states, dtype=torch.float32)
                dones_t = torch.tensor(dones, dtype=torch.float32)

                # get current q value predictions for each action
                q_values = policy_net(states_t)
                q_vals_action = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1)

                # select next actions for each state using existing DQN
                next_q_policy = policy_net(next_states_t)
                next_actions = next_q_policy.argmax(dim=1).unsqueeze(1)

                # get expected Q value for taking those actions from those states
                next_q_target = target_net(next_states_t)
                next_q_vals = next_q_target.gather(1, next_actions).squeeze(1)

                # calculate what the model should be trying to predict
                targets = rewards_t + gamma * (1 - dones_t) * next_q_vals

                # calculate loss
                loss = F.mse_loss(q_vals_action, targets.detach())

                # update gradients
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            episode_length += 1

        # epsilon update
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # clone policy_net into target_net
        if episode % target_update_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % 200 == 0:
            print(f"Episode {episode} | epsilon={epsilon:.3f}")


    return policy_net


game = Quoridor(GUI=False)#True, sleep=0.01)   # Replace with your real environment
trained_net = train_dqn(game)