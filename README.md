# quoridor-ai
CS4100 Final Project
by Timothy Clay, Kim-Cuong Tran Dang, Carrie Wang, and Phineas Wormser

## Requirements
To run this repo, the following installations are required:

- torch~=2.9.1
- numpy~=1.26.3
- tqdm~=4.66.2
- pygame~=2.6.1

### Installations
1. Clone this repo, run in terminal:
```
git clone https://github.com/timothy-clay/quoridor-ai.git
```
2. Go to project directory:
```
cd <your-project-directory>
```
3. Install the dependencies from requirements.txt
```
pip install -r requirements.txt
```

## Descriptions
In this repo, we have implemented the gameplay for quoridor, AI agents to play the game using Minimax and DQN, and
evaluations for the agents
- Core implementation of quoridor gameplay: quoridor.py, fence.py, grid.py, player.py
- agents.py: set up for baseline(minimax depth 1) player, minimax agent, and DQN agents to take turns
- dqn.py, dqn_training: CNN implementation and training
- minimax.py: heuristic + alpha/beta pruning for minimax agent
- play_vs_agent.py: set up gameplay for human player v.s. Minimax or DQN agents, or Minimax v.s. DQN agent (with GUI)
- eval.py: pit DQN v.s. Minimax agent, Minimax v.s. Baseline, DQN v.s. Baseline for input number of games (no GUI)

## Functionality
- To play a game of Quoridor (where you control both players), run the following script in the terminal: `python quoridor.py`
- To play a game of Quoridor against a DQN player (with you as Player 1), run the following script in the terminal: `python play_vs_agent.py`
  - To instead play as Player 1 or play against a minimax opponent, edit the play_vs_agent.py file accordingly (uncomment the relevant lines in main).
- To re-execute our agent evaluations, run the following script in the terminal: `python eval.py`
  - NOTE: this will likely take upwards of 1.5 hours to run
