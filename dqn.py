import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        
        self.main = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        output = self.main(x)
        return output
    
    
