# quoridor-ai
CS4100 Final Project
by Timothy Clay, Kim-Cuong Tran Dang, Carrie Wang, and Phineas Wormser

Presentation Slides: https://docs.google.com/presentation/d/1ht1_yU8qKHoT-HI3Q_t2kZDAi0r3wTMm8lt3t33N4EU/edit?usp=sharing

Final Report: https://docs.google.com/document/d/1Q4xvP06P6Xvs-cEoUP_RhAA6qIq2qXbyuLGsWCSJV6k/edit?usp=sharing

## Descriptions
In this repo, we have implemented the gameplay for quoridor, AI agents to play the game using Minimax and DQN, and
evaluations for the agents
- Core implementation of quoridor gameplay: quoridor.py, fence.py, grid.py, player.py
- agents.py: set up for baseline(minimax depth 1) player, minimax agent, and DQN agents to take turns
- dqn.py, dqn_training: CNN implementation and training
- minimax.py: heuristic + alpha/beta pruning for minimax agent
- play_vs_agent.py: set up gameplay for human player v.s. Minimax or DQN agents, or Minimax v.s. DQN agent (with GUI)
- eval.py: pit DQN v.s. Minimax agent, Minimax v.s. Baseline, DQN v.s. Baseline for input number of games (no GUI)

To execute DQN v.s Minimax agent (depth 2) with GUI, run in terminal:
```
python3 play_vs_agent.py
```

To execute agent evaluations, run in terminal:
```
python3 eval.py
```
