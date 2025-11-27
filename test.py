from quoridor import *
from dqn import *
import numpy as np
import torch
import torch.optim as optim
from collections import deque
import random
from tqdm import tqdm
from time import sleep


if __name__ == "__main__":

    game = Quoridor(True, print_messages = True, sleep=0.1, gs=9)
    winner, grid, players, reward = game.execute('e')

    while True:
        command = input()
        winner, grid, players, reward = game.execute(command)

        print(reward)
        print(f"Paths: P1: {players['player1'].getShortestPath(grid, checkOpponent=True)} | P2: {players['player2'].getShortestPath(grid, checkOpponent=True)}")