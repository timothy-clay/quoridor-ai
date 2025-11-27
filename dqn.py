import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, num_actions):
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
            nn.Linear(128 * 9 * 9 + 5, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, state, prev_move_onehot):

        conv_out = self.conv(state)
        conv_flat = conv_out.reshape(state.size(0), -1)

        combined = torch.cat([conv_flat, prev_move_onehot], dim=1)

        q = self.fc(combined)
        
        return q

