from quoridor import *
from dqn import * 
from minimax import *
import numpy as np
import torch
import torch.functional as F
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm


class PrevMoves:
    """
    Note: this class was created with the help of generative AI (LLMs)
    """

    def __init__(self, capacity=100000):
        """
        Creates a deque with specified capacity to store previous moves taken by an agent.
        """
        self.moves = deque(maxlen=capacity)

    def add(self, s, pmo, a, r, s2, pmo2, d):
        """
        Store a new sequence to the list of moves. Arguments refer to state, previous move, action, reward, next state, 
        next previous action, and a done indicator.
        """
        self.moves.append((s, pmo, a, r, s2, pmo2, d))

    def sample(self, batch_size):
        """
        Randomly sample a batch size from the stored previous moves.
        """
        batch = random.sample(self.moves, batch_size)
        s, pmo, a, r, s2, pmo2, done = map(np.array, zip(*batch))
        return s, pmo, a, r, s2, pmo2, done

    def __len__(self):
        return len(self.moves)
    

def dqn_epsilon_greedy(dqn, grid, players, game, epsilon, device):
    """
    Take an action with a DQN agent according to a pseudo-epsilon-greedy approach. 
    """

    # take random move with probability epsilon
    if np.random.random() < epsilon:

        # get a list of all movement and fence indices
        movement_actions = list(range(4))
        fence_actions = list(range(4, len(ALL_ACTIONS)))

        # shuffle the fence indices
        random.shuffle(fence_actions)

        # combine movement options and first four fence options, then shuffle those options
        first_block = movement_actions + fence_actions[:4]
        random.shuffle(first_block)  

        # get the remaining fence indices that do not appear in the first block
        remaining_fences = fence_actions[4:]

        # combine the first block of candidate actions with the remaining fences
        candidate_actions = first_block + remaining_fences
    
    else:

        # get the current state and one-hot previous move for the player and convert to tensors
        state, prev_move_onehot = game.getState(grid, players)
        s_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        pmo_tensor = torch.tensor(prev_move_onehot, dtype=torch.float32, device=device).unsqueeze(0)

        # get the q values for the given state
        q_values = dqn(s_tensor, pmo_tensor)[0].detach().cpu().numpy()

        # sort the action indices in descending order of value
        candidate_actions = np.argsort(q_values)[::-1]

    # loop through each candidate action, check if it's valid, and return its index if so
    for candidate_action in candidate_actions:
        if players['active_player'].checkMoveValidity(game, grid, players['inactive_player'], ALL_ACTIONS[candidate_action]):
            return candidate_action
        
    return None


def calc_loss(policy_net, target_net, states, prev_move_onehots, actions, rewards, next_states, next_prev_move_onehots, dones, gamma, device):
    """
    Calculates the MSE loss for a given policy network and a batch of previous moves. 

    Note: this function was created with the assistance of generative AI (LLMs)
    """

    # convert inputs to tensors
    states_t = torch.tensor(states, dtype=torch.float32, device=device)
    prev_move_onehots_t = torch.tensor(prev_move_onehots, dtype=torch.float32, device=device)
    next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device)
    next_prev_move_onehots_t = torch.tensor(next_prev_move_onehots, dtype=torch.float32, device=device)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

    # get the q values for the DQN
    q_values = policy_net(states_t, prev_move_onehots_t)
    q_a = q_values.gather(1, actions_t.unsqueeze(1)).squeeze(1) 

    # calculate the target value for the DQN
    with torch.no_grad():
        next_q_policy = policy_net(next_states_t, next_prev_move_onehots_t) 
        next_actions = next_q_policy.argmax(dim=1, keepdim=True)
        next_q_target = target_net(next_states_t, next_prev_move_onehots_t) 
        next_q_target_selected = next_q_target.gather(1, next_actions).squeeze(1) 
        target = rewards_t + gamma * (1.0 - dones_t) * next_q_target_selected

    # clamps the target to avoid runaway targets
    target = torch.clamp(target, -15.0, 20.0)

    # calculate and return MSE loss
    loss = F.mse_loss(q_a, target)
    return loss


def evaluate(game, P1, P2, games=100, device='cpu'):
    """ 
    Simulate a series of games in which two DQN players play against each other in evaluation mode. 
    """
    
    # initialize performance metrics
    wins_P1, wins_P2 = 0, 0
    rewards_P1, rewards_P2 = [], []

    turns = 0
    
    for _ in tqdm(range(games)):

        # reset game state
        game.reset()

        # get current state
        winner, grid, players, reward = game.execute('e')
        state, prev_move_onehot = game.getState(grid, players)

        # define starting player
        if players['active_player'] == players['player1']:
            current_player = 1

        # track cumulative P1 and P2 rewards
        reward_P1, reward_P2 = 0, 0

        # loop until there's a winner or the game has gone on for 250 turns
        game_length = 0
        while winner is None and game_length < 250:

            # take turn according to current player (with 5% chance of a random move)
            if current_player == 1:
                a = dqn_epsilon_greedy(P1, grid, players, game, epsilon=0.05, device=device)
            else:
                a = dqn_epsilon_greedy(P2, grid, players, game, epsilon=0.05, device=device)

            try:

                # take selected move
                winner, grid, players, reward = game.execute(ALL_ACTIONS[a])

                # update reward for current player
                if current_player == 1:
                    reward_P1 += reward
                else:
                    reward_P2 += reward

                # alternate players and iterate game length
                current_player = 3 - current_player
                game_length += 1
            
            # catch invalid moves (bugs in implementation) and end game
            except TypeError:
                break
        
        # record game winner
        if winner == players['player1']:
            wins_P1 += 1
        elif winner == players['player2']:
            wins_P2 += 1

        # store cumulative rewards
        rewards_P1.append(reward_P1)
        rewards_P2.append(reward_P2)

        # update total turns
        turns += game_length
    
    # compute and return evaluation metrics
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
    """ 
    Visualize two DQN players playing one game against each other in evaluation mode. 
    """

    game = Quoridor(GUI=True, sleep=0.5)

    # define starting player
    if players['active_player'] == players['player1']:
        current_player = 1

    # loop until there's a winner or the game has gone on for 250 turns
    game_length = 0
    while winner is None and game_length < 250:

        # take turn according to current player (with 5% chance of a random move)
        try:
            if current_player == 1:
                a = dqn_epsilon_greedy(P1, grid, players, game, epsilon=0.05, device=device, verbose=False)
            else:
                a = dqn_epsilon_greedy(P2, grid, players, game, epsilon=0.05, device=device, verbose=False)

            winner, grid, players, reward = game.execute(ALL_ACTIONS[a])

            current_player = 3 - current_player
            game_length += 1
        
        # catch invalid moves (bugs in implementation) and end game
        except TypeError:
            break

    pygame.quit()




def train(game, epochs = 10, train_episodes=250, batch_size=64, gamma=0.6, lr_p1=1e-4, lr_p2=1e-4, epsilon_decay=0.998):
    """ 
    Train two DQN agents by playing 5,000 games against each other and alternating training periods. 

    Note: this function was created with the assistance of generative AI (LLMs) in select parts. 
    """
    
    # use GPU if possible
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    # create policy and target networks (Player 1)
    n_actions = game.num_actions
    p1_policy_net = DQN(n_actions).to(device)
    p1_target_net = DQN(n_actions).to(device)
    p1_target_net.load_state_dict(p1_policy_net.state_dict())
    p1_target_net.eval()

    # create policy and target networks (Player 1)
    p2_policy_net = DQN(n_actions).to(device)
    p2_target_net = DQN(n_actions).to(device)
    p2_target_net.load_state_dict(p2_policy_net.state_dict())
    p2_target_net.eval()

    # define optimizers
    opt_p1 = optim.Adam(p1_policy_net.parameters(), lr=lr_p1)
    opt_p2 = optim.Adam(p2_policy_net.parameters(), lr=lr_p2)

    # create custom deques to store agent moves
    p1_moves = PrevMoves()
    p2_moves = PrevMoves()

    # initialize model checkpoints
    p1_init = DQN(n_actions).to(device) 
    p1_init.load_state_dict(p1_policy_net.state_dict()) 
    p1_init.eval() 
    p1_checkpoints = [p1_init]
    
    p2_init = DQN(n_actions).to(device) 
    p2_init.load_state_dict(p2_policy_net.state_dict()) 
    p2_init.eval() 
    p2_checkpoints = [p2_init]
    
    # initialize epsilon values
    p1_epsilon = 1
    p2_epsilon = 1

    # track total number of turns taken
    total_steps = 0

    # loop through each training epoch
    for epoch in range(epochs):
        
        # alternate between players
        for train_agent in [1, 2]:

            # simulate training games for the current player
            for episode in tqdm(range(train_episodes)):

                # randomly select opponent agents to play against (so as to learn against several different opponents)
                agent_probs = [0.5 ** i for i in range(len(p1_checkpoints))][::-1]
                agent_probs_norm = [p / sum(agent_probs) for p in agent_probs]
                p1_agent = np.random.choice(p1_checkpoints, p=agent_probs_norm)
                p2_agent = np.random.choice(p2_checkpoints, p=agent_probs_norm)

                # reset game state
                game.reset()

                # get current state
                winner, grid, players, reward = game.execute('e')
                state, prev_move_onehot = game.getState(grid, players)
                done = False

                # define starting player
                if players['active_player'] == players['player1']:
                    current_player = 1

                # take turns until there's a winner or the game has gone on for 250 turns
                episode_length = 0
                while not done and episode_length < 250:
                    
                    # choose action according to the current player and the learning player
                    if current_player == 1:
                        if train_agent == 1:
                            action = dqn_epsilon_greedy(p1_policy_net, grid, players, game, p1_epsilon, device)
                        else:
                            action = dqn_epsilon_greedy(p1_agent, grid, players, game, 0.01, device)
                    else:
                        if train_agent == 2:
                            action = dqn_epsilon_greedy(p2_policy_net, grid, players, game, p2_epsilon, device)
                        else:
                            action = dqn_epsilon_greedy(p2_agent, grid, players, game, 0.01, device)

                    # (try to) take action and get next state
                    try:
                        winner, grid, players, reward = game.execute(ALL_ACTIONS[action])
                        next_state, next_prev_move_onehot = game.getState(grid, players)
            
                    # catch invalid moves (bugs in implementation) and end game
                    except TypeError:
                        done = True
                        continue

                    # end game if there's a winner
                    if winner is not None:
                        done = True
                
                    # add move to the stored moves according to the current player and the learning player
                    if current_player == 1 and train_agent == 1:
                        p1_moves.add(state, prev_move_onehot, action, reward, next_state, next_prev_move_onehot, float(done))
                    elif current_player == 2 and train_agent == 2:
                        p2_moves.add(state, prev_move_onehot, action, reward, next_state, next_prev_move_onehot, float(done))

                    # update state and prev_move_onehot
                    state, prev_move_onehot = next_state, next_prev_move_onehot
                    
                    # increment episode length and total number of steps
                    episode_length += 1
                    total_steps += 1

                    # update weights every 4 turns
                    if total_steps % 4 == 0:

                        # update P1 weights if P1 is learning agent
                        if len(p1_moves) >= 2000 and train_agent == 1:
                            s_b, pmo_b, a_b, r_b, s2_b, pmo2_b, done_b = p1_moves.sample(batch_size)
                            loss_p1 = calc_loss(p1_policy_net, p1_target_net, s_b, pmo_b, a_b, r_b, s2_b, pmo2_b, done_b, gamma, device)
                            opt_p1.zero_grad()
                            loss_p1.backward()
                            torch.nn.utils.clip_grad_norm_(p1_policy_net.parameters(), 1.0)
                            opt_p1.step()

                        # update P2 weights if P2 is learning agent
                        if len(p2_moves) >= 2000 and train_agent == 2:
                            s_b, pmo_b, a_b, r_b, s2_b, pmo2_b, done_b = p2_moves.sample(batch_size)
                            loss_p2 = calc_loss(p2_policy_net, p2_target_net, s_b, pmo_b, a_b, r_b, s2_b, pmo2_b, done_b, gamma, device)
                            opt_p2.zero_grad()
                            loss_p2.backward()
                            torch.nn.utils.clip_grad_norm_(p2_policy_net.parameters(), 1.0)
                            opt_p2.step()

                    # alternate players
                    current_player = 3 - current_player

                # update epsilon for learning player
                if train_agent == 1:
                    p1_epsilon = max(0.05, p1_epsilon * epsilon_decay)
                else:
                    p2_epsilon = max(0.05, p2_epsilon * epsilon_decay)

        # update target networks after each epoch
        p1_target_net.load_state_dict(p1_policy_net.state_dict())
        p2_target_net.load_state_dict(p2_policy_net.state_dict())

        # store current P1 model after each epoch for use in later training cycles
        p1_checkpoint = DQN(n_actions).to(device) 
        p1_checkpoint.load_state_dict(p1_policy_net.state_dict()) 
        p1_checkpoint.eval() 
        p1_checkpoints.append(p1_checkpoint)

        # store current P2 model after each epoch for use in later training cycles
        p2_checkpoint = DQN(n_actions).to(device) 
        p2_checkpoint.load_state_dict(p2_policy_net.state_dict()) 
        p2_checkpoint.eval() 
        p2_checkpoints.append(p2_checkpoint)

        # evaluate head-to-head performance after each epoch, print results, and visualize one game
        eval_results = evaluate(game, p1_policy_net, p2_policy_net, games=100, device=device)
        print(f"P1 Win %: {eval_results['P1_winrate']:.2f} | P2 Win %: {eval_results['P2_winrate']:.2f} | Avg. Game Length: {eval_results['avg_game_length']:.0f}    \nAvg. P1 Reward: {eval_results['avg_p1_reward']:.2f} | Med. P1 Reward: {eval_results['med_p1_reward']:.2f}\nAvg. P2 Reward: {eval_results['avg_p2_reward']:.2f} | Med. P2 Reward: {eval_results['med_p2_reward']:.2f}")
        visualize_game(p1_policy_net, p2_policy_net, device=device)
            
    # save the optimal policies after 10 epochs of training
    torch.save(p1_policy_net.state_dict(), 'p1_policy.pth')
    torch.save(p2_policy_net.state_dict(), 'p2_policy.pth')

    return


if __name__=="__main__":
    game = Quoridor(GUI=False)
    train(game)