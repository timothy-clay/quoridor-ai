import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 9 * 9, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        features = self.conv(x)
        flat = features.reshape(x.size(0), -1)
        q = self.fc(flat)
        return q
