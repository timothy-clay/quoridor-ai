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
This repo contains code to create an implementation of Quoridor from scratch, build and train minimax and DQN agents to intelligently play Quoridor, and evaluate their performance against human players and other agents. The following outline the contents of each individual file:
- Game Implementation
  - `quoridor.py`: contains main implementation of the game and several helper functions related to the game and game state
  - `grid.py`: contains a grid class that manages everything that happens on the grid of the board (like fence placements)
  - `player.py`: contains a player class that manages everything that a player could do (like move)
  - `fence.py`: contains a fence class that stores minimal information about each placed fence, such as coordinates
- Agent Creation / Training
  - `minimax.py`: contains our heuristic function and minimax function that dictate how a minimax player behaves
  - `dqn.py`: contains our architecture definition for all of our DQN agents
  - `dqn_training.py`: contains our DQN training loop and several helper functions that assist in training
- Agent Evaluation / Gameplay
  - `agents.py`: contains agent classes and subclasses that allow universal syntax for taking a turn
  - `play_vs_agent.py`: contains a `play_vs_agent()` function that allows users to play against one of the trained agents
  - `eval.py`: contains functions to simulate 100 games of various matchups (e.g., DQN vs. minimax)

## Functionality
- To play a game of Quoridor (where you control both players), run the following script in the terminal: `python quoridor.py`
- To play a game of Quoridor against a DQN player (with you as Player 1), run the following script in the terminal: `python play_vs_agent.py`
  - To instead play as Player 1 or play against a minimax opponent, edit the play_vs_agent.py file accordingly (uncomment the relevant lines in main).
- To re-execute our agent evaluations, run the following script in the terminal: `python eval.py`
  - NOTE: this will likely take upwards of 1.5 hours to run
